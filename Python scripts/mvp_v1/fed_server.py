# fed_server.py  –  Federated Learning IDS Server (Windows)
"""
Central aggregation server. Runs on Windows alongside the MQTT broker.

Responsibilities:
  1. Wait for min_clients Pi devices to connect and send GRU weight arrays
  2. Run FedAvg (weighted average of weight arrays by num_samples)
  3. Push the averaged numpy weights back to all Pi clients via MQTT
  4. Repeat every federation round (triggered by client updates OR adaptive schedule)
  5. Log per-layer weight norms and divergence statistics for each round

Architecture match (must stay in sync with fed_client.py):
  Model  : GRU seq2seq autoencoder — pure numpy on Pi, numpy FedAvg on server
  Weights: 16 named arrays (encoder GRU x7, decoder GRU x7, output proj x2)

  F = N_FEATURES = 8   H = GRU_HIDDEN = 32   xh = F+H = 40

  Key          Shape       Description
  ──────────── ─────────── ──────────────────────────────────────
  Wr           (40, 32)    encoder reset gate weight
  br           (32,)       encoder reset gate bias
  Wz           (40, 32)    encoder update gate weight
  bz           (32,)       encoder update gate bias
  Wn_x         ( 8, 32)    encoder new gate input weight
  Wn_h         (32, 32)    encoder new gate hidden weight
  bn           (32,)       encoder new gate bias
  Dec_Wr       (40, 32)    decoder reset gate weight
  Dec_br       (32,)       decoder reset gate bias
  Dec_Wz       (40, 32)    decoder update gate weight
  Dec_bz       (32,)       decoder update gate bias
  Dec_Wn_x     ( 8, 32)    decoder new gate input weight
  Dec_Wn_h     (32, 32)    decoder new gate hidden weight
  Dec_bn       (32,)       decoder new gate bias
  Wo           (32,  8)    output projection weight
  bo           ( 8,)       output projection bias

The server NEVER sees raw CAN data — only:
  - Weight arrays (list of 16 numpy ndarrays)
  - num_samples  (int, for weighted averaging)
  - loss         (float, for monitoring)

Requirements (Windows server):
  pip install paho-mqtt numpy
"""

import os
import json
import time
import pickle
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict
import paho.mqtt.client as mqtt

# ── CONFIG ────────────────────────────────────────────────────────────────────
MQTT_BROKER    = "localhost"
MQTT_PORT      = 1883
MIN_CLIENTS    = 2           # minimum updates before triggering FedAvg

# ── ADAPTIVE ROUND SCHEDULE ──────────────────────────────────────────────────
# Vehicle battery lasts ~14 hours. Instead of a single 6-hour fallback,
# we use an adaptive two-phase schedule:
#   Early phase  : aggregate every 15 min → rapid model convergence
#   Late phase   : aggregate every 60 min → reduced overhead once stable
# Transition occurs when avg loss improvement between consecutive rounds
# falls below LOSS_IMPROVEMENT_THRESHOLD (i.e. model has largely converged).
EARLY_INTERVAL_SEC         = 900     # 15 minutes
LATE_INTERVAL_SEC          = 3600    # 60 minutes
LOSS_IMPROVEMENT_THRESHOLD = 0.05    # 5% relative improvement triggers switch

CHECKPOINT_DIR = "./models/checkpoints"

# ── GRU architecture constants (must match fed_client.py exactly) ─────────────
N_FEATURES  = 8
GRU_HIDDEN  = 32
WINDOW_SIZE = 20
INPUT_DIM   = WINDOW_SIZE * N_FEATURES   # 160

# Ordered weight keys and their expected shapes — used for validation and logging
# Any mismatch between this table and received arrays is caught before aggregation.
_F, _H, _XH = N_FEATURES, GRU_HIDDEN, N_FEATURES + GRU_HIDDEN   # 8, 32, 40

WEIGHT_SCHEMA = [
    # key            expected shape    gate / component
    ("Wr",           (_XH, _H)),      # encoder reset
    ("br",           (_H,)),
    ("Wz",           (_XH, _H)),      # encoder update
    ("bz",           (_H,)),
    ("Wn_x",         (_F,  _H)),      # encoder new (input path)
    ("Wn_h",         (_H,  _H)),      # encoder new (hidden path)
    ("bn",           (_H,)),
    ("Dec_Wr",       (_XH, _H)),      # decoder reset
    ("Dec_br",       (_H,)),
    ("Dec_Wz",       (_XH, _H)),      # decoder update
    ("Dec_bz",       (_H,)),
    ("Dec_Wn_x",     (_F,  _H)),      # decoder new (input path)
    ("Dec_Wn_h",     (_H,  _H)),      # decoder new (hidden path)
    ("Dec_bn",       (_H,)),
    ("Wo",           (_H,  _F)),      # output projection
    ("bo",           (_F,)),
]
WEIGHT_KEYS   = [k for k, _ in WEIGHT_SCHEMA]
WEIGHT_SHAPES = {k: s for k, s in WEIGHT_SCHEMA}
N_WEIGHTS     = len(WEIGHT_SCHEMA)   # must be 16
# ─────────────────────────────────────────────────────────────────────────────


def _validate_weights(weights: list, client_id: str) -> bool:
    """
    Check that received weight list matches the GRU schema exactly.
    Returns True if valid, False (and prints reason) if not.
    """
    if len(weights) != N_WEIGHTS:
        print(f"[Validator] {client_id}: expected {N_WEIGHTS} weight arrays, "
              f"got {len(weights)} — skipping this update")
        return False

    for i, ((key, expected_shape), arr) in enumerate(
            zip(WEIGHT_SCHEMA, weights)):
        if not isinstance(arr, np.ndarray):
            print(f"[Validator] {client_id}: weight[{i}] ({key}) is not "
                  f"ndarray (got {type(arr)}) — skipping")
            return False
        if arr.shape != expected_shape:
            print(f"[Validator] {client_id}: weight[{i}] ({key}) shape "
                  f"mismatch — expected {expected_shape}, got {arr.shape} "
                  f"— skipping")
            return False

    return True


class FedAvgAggregator:
    """
    Weighted FedAvg over lists of 16 numpy GRU weight arrays.

    Each client sends:
      {'weights': [np.ndarray x16], 'num_samples': int,
       'loss': float, 'round': int}

    Aggregation:
      averaged[i] = Σ_k  (n_k / N_total) × weights_k[i]

    After aggregation, per-layer statistics are logged:
      - L2 norm of the averaged weight
      - L2 norm from each contributing client (divergence indicator)
    """

    def __init__(self):
        self.updates = {}       # client_id → {weights, num_samples, loss, round}
        self.lock    = threading.Lock()

    def add_update(self, client_id: str, weights: list,
                   num_samples: int, loss: float, round_num: int) -> bool:
        """Validate and store a client weight update. Returns True if accepted."""
        if not _validate_weights(weights, client_id):
            return False
        with self.lock:
            self.updates[client_id] = {
                "weights":     [w.astype(np.float32) for w in weights],
                "num_samples": num_samples,
                "loss":        loss,
                "round":       round_num,
                "timestamp":   time.time(),
            }
        print(f"[FedAvg] Accepted update from {client_id}: "
              f"samples={num_samples}, loss={loss:.6f}, round={round_num}")
        return True

    def n_updates(self) -> int:
        with self.lock:
            return len(self.updates)

    def aggregate(self) -> tuple:
        """
        Weighted average of all accepted weight arrays.
        Returns (list of 16 np.float32 ndarrays, avg_loss float),
        or (None, None) if no updates.
        Logs per-layer norm statistics for research transparency.
        """
        with self.lock:
            if not self.updates:
                return None, None

            total_samples = sum(u["num_samples"]
                                for u in self.updates.values())
            if total_samples == 0:
                return None, None

            client_ids = list(self.updates.keys())
            losses     = [self.updates[c]["loss"] for c in client_ids]
            avg_loss   = float(np.mean(losses))

            # ── Weighted average ──────────────────────────────────────────
            averaged = None
            for update in self.updates.values():
                frac   = update["num_samples"] / total_samples
                params = update["weights"]
                if averaged is None:
                    averaged = [w * frac for w in params]
                else:
                    for i, w in enumerate(params):
                        averaged[i] = averaged[i] + w * frac

            # ── Per-layer divergence logging ──────────────────────────────
            print(f"\n[FedAvg] Round summary ── "
                  f"{len(self.updates)} client(s) | "
                  f"total_samples={total_samples} | "
                  f"avg_loss={avg_loss:.6f} | "
                  f"min_loss={np.min(losses):.6f}")
            print(f"{'Layer':<14} {'Avg norm':>10} "
                  f"{'Client norms (per client)':>30}")
            print("─" * 60)

            for i, (key, _) in enumerate(WEIGHT_SCHEMA):
                avg_norm     = float(np.linalg.norm(averaged[i]))
                client_norms = [
                    float(np.linalg.norm(
                        self.updates[c]["weights"][i]))
                    for c in client_ids
                ]
                norm_strs = "  ".join(f"{n:.4f}" for n in client_norms)
                print(f"  {key:<12} {avg_norm:>10.4f}   {norm_strs}")

            print("─" * 60)

            # ── Weight drift: mean absolute deviation across clients ───────
            drifts = []
            for i in range(N_WEIGHTS):
                stacked = np.stack(
                    [self.updates[c]["weights"][i].flatten()
                     for c in client_ids], axis=0)
                drift = float(np.mean(np.std(stacked, axis=0)))
                drifts.append(drift)
            top3 = sorted(
                zip([k for k, _ in WEIGHT_SCHEMA], drifts),
                key=lambda x: -x[1])[:3]
            print(f"[FedAvg] Top-3 drifting layers: "
                  + ", ".join(f"{k}={d:.5f}" for k, d in top3))

            self.updates.clear()
            return averaged, avg_loss


class WeightCheckpointer:
    """
    Saves averaged weight arrays as .npz files after each round.
    Provides a load method so the server can resume from a checkpoint
    and broadcast to late-joining clients.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.last_averaged  = None   # most recent averaged weights (list)
        self.last_round     = 0

    def save(self, averaged_weights: list, round_num: int) -> str:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.checkpoint_dir,
                            f"global_round{round_num}_{ts}.npz")
        # Save with named keys for inspectability
        save_dict = {key: arr for key, arr
                     in zip(WEIGHT_KEYS, averaged_weights)}
        np.savez(path, **save_dict)
        size_kb = sum(w.nbytes for w in averaged_weights) / 1024
        print(f"[Checkpoint] Saved round {round_num}: {path} "
              f"({size_kb:.1f} KB, {N_WEIGHTS} weight arrays)")
        self.last_averaged = averaged_weights
        self.last_round    = round_num
        return path

    def load_latest(self) -> list:
        """Load the most recent checkpoint from disk, if any."""
        files = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("global_round") and f.endswith(".npz")
        ])
        if not files:
            return None
        path = os.path.join(self.checkpoint_dir, files[-1])
        try:
            d = np.load(path)
            weights = [d[k].astype(np.float32) for k in WEIGHT_KEYS]
            print(f"[Checkpoint] Loaded latest checkpoint: {path}")
            return weights
        except Exception as e:
            print(f"[Checkpoint] Failed to load {path}: {e}")
            return None


class FederatedServer:
    """
    Main server: receives GRU weight updates, runs FedAvg,
    pushes averaged weights back to Pi clients over MQTT.

    Adaptive scheduling:
      - Early phase (first ~2h): round every 15 min → ~8 rounds
      - Late phase  (remaining): round every 60 min → ~12 rounds
      - Total: ~20 rounds over a 14-hour vehicle battery life
      - Phase transition: when loss improvement < 5% between rounds
    """

    def __init__(self, broker=MQTT_BROKER, port=MQTT_PORT,
                 min_clients=MIN_CLIENTS):
        self.broker      = broker
        self.port        = port
        self.min_clients = min_clients

        self.aggregator   = FedAvgAggregator()
        self.checkpointer = WeightCheckpointer(CHECKPOINT_DIR)

        self.mqtt_client        = None
        self.connected          = False
        self.current_round      = 0
        self.round_start        = time.time()
        self.known_clients      = set()
        self._round_lock        = threading.Lock()
        self._round_in_progress = False

        # ── Adaptive scheduling state ─────────────────────────────────────
        self.round_interval = EARLY_INTERVAL_SEC   # start in fast mode
        self.phase          = "early"               # "early" or "late"
        self.loss_history   = []                    # avg_loss per round

    # ── MQTT ──────────────────────────────────────────────────────────────────

    def start_mqtt(self):
        try:
            self.mqtt_client = mqtt.Client(
                client_id="fed_server",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        except Exception:
            self.mqtt_client = mqtt.Client(client_id="fed_server")

        self.mqtt_client.on_connect    = self._on_connect
        self.mqtt_client.on_message    = self._on_message
        self.mqtt_client.on_disconnect = self._on_disconnect

        print(f"Connecting to MQTT broker {self.broker}:{self.port}...")
        self.mqtt_client.connect(self.broker, self.port, keepalive=60)
        self.mqtt_client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if hasattr(rc, 'value'):
            rc = rc.value
        if rc == 0:
            self.connected = True
            print("Server connected to MQTT broker")
            client.subscribe("federated/model/updates")
            client.subscribe("federated/clients/status")

            # Re-broadcast the last global model to any clients that
            # connected while the server was offline
            if self.checkpointer.last_averaged is not None:
                print("[Server] Re-broadcasting last global model "
                      "to reconnected clients...")
                self._broadcast_global_model(
                    self.checkpointer.last_averaged)
        else:
            print(f"Server MQTT connection failed: rc={rc}")

    def _on_disconnect(self, client, userdata, disconnect_flags,
                       rc, properties=None):
        self.connected = False
        print("Server disconnected from MQTT broker")

    def _on_message(self, client, userdata, message):
        topic = message.topic
        try:
            if topic == "federated/model/updates":
                self._handle_weight_update(message.payload)
            elif topic == "federated/clients/status":
                status = json.loads(message.payload)
                cid    = status.get("client_id")
                if cid:
                    self.known_clients.add(cid)
        except Exception as e:
            print(f"[Server] Message error on {topic}: {e}")

    def _handle_weight_update(self, payload: bytes):
        """Parse, validate, and register an incoming GRU weight update."""
        try:
            update = pickle.loads(payload)

            client_id   = update["client_id"]
            weights     = update["weights"]       # list of 16 np.ndarray
            num_samples = update["num_samples"]
            loss        = update["loss"]
            round_num   = update.get("round", self.current_round)

            accepted = self.aggregator.add_update(
                client_id, weights, num_samples, loss, round_num)

            if not accepted:
                return

            n = self.aggregator.n_updates()
            print(f"[Server] Updates collected: {n}/{self.min_clients}")

            if n >= self.min_clients and not self._round_in_progress:
                threading.Thread(
                    target=self._run_fedavg_round, daemon=True).start()

        except Exception as e:
            print(f"[Server] Weight update parse error: {e}")

    # ── FedAvg round ──────────────────────────────────────────────────────────

    def _run_fedavg_round(self):
        with self._round_lock:
            if self._round_in_progress:
                return
            self._round_in_progress = True

        try:
            self.current_round += 1
            print(f"\n{'='*50}")
            print(f"[Server] Starting FedAvg round {self.current_round}")
            print(f"[Server] Phase: {self.phase.upper()} "
                  f"(interval: {self.round_interval/60:.0f} min)")
            print(f"{'='*50}")

            averaged_weights, avg_loss = self.aggregator.aggregate()
            if averaged_weights is None:
                print("[Server] No updates to aggregate")
                return

            # Verify the averaged result before saving/broadcasting
            if not _validate_weights(averaged_weights, "averaged_result"):
                print("[Server] Averaged weights failed validation — "
                      "round aborted")
                return

            # Save checkpoint
            self.checkpointer.save(averaged_weights, self.current_round)

            # ── Adaptive phase switching ──────────────────────────────────
            # Track loss and switch from early→late when improvement stalls
            self.loss_history.append(avg_loss)
            if (self.phase == "early"
                    and len(self.loss_history) >= 2
                    and self.loss_history[-2] > 0):
                prev_loss = self.loss_history[-2]
                improvement = (prev_loss - avg_loss) / prev_loss

                print(f"[Scheduler] Loss: {avg_loss:.6f} | "
                      f"Prev: {prev_loss:.6f} | "
                      f"Improvement: {improvement:.2%}")

                if improvement < LOSS_IMPROVEMENT_THRESHOLD:
                    self.phase = "late"
                    self.round_interval = LATE_INTERVAL_SEC
                    print(f"[Scheduler] *** PHASE SWITCH: early → late ***")
                    print(f"[Scheduler] Loss improvement {improvement:.2%} "
                          f"< {LOSS_IMPROVEMENT_THRESHOLD:.0%} threshold")
                    print(f"[Scheduler] Round interval now "
                          f"{LATE_INTERVAL_SEC/60:.0f} min")
            # ──────────────────────────────────────────────────────────────

            # Broadcast to all Pi clients
            self._broadcast_global_model(averaged_weights)

            self.round_start = time.time()
            print(f"[Server] Round {self.current_round} complete | "
                  f"Next round in ~{self.round_interval/60:.0f} min\n")

        except Exception as e:
            print(f"[Server] FedAvg round error: {e}")
        finally:
            self._round_in_progress = False

    def _broadcast_global_model(self, averaged_weights: list):
        """
        Publish averaged GRU weight arrays to all Pi clients.

        Payload keys (must match fed_client.py _apply_global_model):
          round          : int
          fedavg_weights : list of 16 np.ndarray  ← client reads this key
          weight_keys    : list of str             ← for client-side sanity check
          timestamp      : float
        """
        if not self.connected:
            print("[Server] Not connected — broadcast skipped")
            return

        payload = pickle.dumps({
            "round":          self.current_round,
            "fedavg_weights": averaged_weights,   # 16 np.float32 arrays
            "weight_keys":    WEIGHT_KEYS,         # key order confirmation
            "timestamp":      time.time(),
        })
        result = self.mqtt_client.publish(
            "federated/model/global", payload, qos=1, retain=True)
        result.wait_for_publish()

        total_kb = sum(w.nbytes for w in averaged_weights) / 1024
        print(f"[Server] Global model broadcast: "
              f"round={self.current_round}, "
              f"arrays={len(averaged_weights)}, "
              f"weights_kb={total_kb:.1f}, "
              f"payload_kb={len(payload)/1024:.1f}")

    # ── Adaptive scheduled round watcher ──────────────────────────────────────

    def _scheduled_round_watcher(self):
        """
        Adaptive federation scheduler — replaces the old 6-hour fallback.

        Checks every 60 seconds whether the current round_interval has
        elapsed. If at least 1 client update is waiting, triggers FedAvg.

        Early phase (~first 2h):  round every 15 min  →  ~8 rounds
        Late  phase (~next 12h):  round every 60 min  →  ~12 rounds
        Total over 14h battery:   ~20 federated rounds
        """
        while True:
            time.sleep(60)
            elapsed = time.time() - self.round_start
            n       = self.aggregator.n_updates()

            if (elapsed >= self.round_interval and n >= 1
                    and not self._round_in_progress):
                print(f"\n[Scheduler] {self.phase.upper()} phase — "
                      f"triggering round after {elapsed/60:.0f} min "
                      f"with {n} update(s)")
                threading.Thread(
                    target=self._run_fedavg_round, daemon=True).start()

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("FEDERATED LEARNING IDS SERVER")
        print(f"  Model          : GRU({GRU_HIDDEN}) seq2seq autoencoder")
        print(f"  Weight arrays  : {N_WEIGHTS} "
              f"(enc x7, dec x7, proj x2)")
        print(f"  Min clients    : {self.min_clients}")
        print(f"  Schedule       : adaptive "
              f"(early={EARLY_INTERVAL_SEC/60:.0f}min, "
              f"late={LATE_INTERVAL_SEC/60:.0f}min)")
        print(f"  Phase switch   : loss improvement "
              f"< {LOSS_IMPROVEMENT_THRESHOLD:.0%}")
        print(f"  MQTT broker    : {self.broker}:{self.port}")
        print(f"  Checkpoints    : {CHECKPOINT_DIR}")
        print("=" * 60)

        # Print weight schema for reference
        print(f"\n{'Layer':<14} {'Shape':>12}   Description")
        print("─" * 50)
        descs = [
            "encoder reset gate", "", "encoder update gate", "",
            "encoder new (input)", "encoder new (hidden)", "",
            "decoder reset gate", "", "decoder update gate", "",
            "decoder new (input)", "decoder new (hidden)", "",
            "output projection", "",
        ]
        for (key, shape), desc in zip(WEIGHT_SCHEMA, descs):
            print(f"  {key:<12} {str(shape):>12}   {desc}")
        print()

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # Try to restore the latest checkpoint so late-joining clients
        # get a warm model immediately on connection
        latest = self.checkpointer.load_latest()
        if latest is not None:
            self.checkpointer.last_averaged = latest

        self.start_mqtt()

        threading.Thread(
            target=self._scheduled_round_watcher, daemon=True).start()

        for _ in range(20):
            if self.connected:
                break
            time.sleep(0.5)

        if not self.connected:
            print("ERROR: Could not connect to MQTT broker. "
                  "Is Mosquitto running?")
            return

        print(f"Server ready. Waiting for {self.min_clients} "
              f"client update(s)...\n")

        try:
            while True:
                n         = self.aggregator.n_updates()
                elapsed_m = (time.time() - self.round_start) / 60
                next_in   = max(0, (self.round_interval / 60) - elapsed_m)
                print(f"  [Round {self.current_round}] "
                      f"Updates: {n}/{self.min_clients} | "
                      f"Clients seen: {len(self.known_clients)} | "
                      f"Phase: {self.phase} | "
                      f"Next in: {next_in:.0f}min",
                      end='\r')
                time.sleep(10)

        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                try:
                    self.mqtt_client.disconnect()
                except Exception:
                    pass


def main():
    import argparse
    p = argparse.ArgumentParser(description="Federated IDS Server")
    p.add_argument("--broker",      default=MQTT_BROKER)
    p.add_argument("--port",        type=int, default=MQTT_PORT)
    p.add_argument("--min-clients", type=int, default=MIN_CLIENTS)
    args = p.parse_args()

    FederatedServer(
        broker=args.broker,
        port=args.port,
        min_clients=args.min_clients,
    ).run()


if __name__ == "__main__":
    main()
