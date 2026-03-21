# fed_client.py  –  Federated Learning Edge Client (Raspberry Pi)
"""
Runs on each Raspberry Pi. Zero PyTorch dependency.

Key design:
  ┌─────────────────────────────────────────────────────┐
  │  PHASE 1 — Bootstrap                                │
  │  Receives trainable .tflite from server via MQTT    │
  │  (untrained model, random weights)                  │
  ├─────────────────────────────────────────────────────┤
  │  PHASE 2 — Local pre-training                       │
  │  Calls tflite 'train' signature on local CAN data   │
  │  until loss stabilises (no server contact)          │
  ├─────────────────────────────────────────────────────┤
  │  PHASE 3 — Federation                               │
  │  Sends weight arrays to server (NOT raw data)       │
  │  Receives averaged global .tflite from server       │
  │  Continues local training on top of global model    │
  ├─────────────────────────────────────────────────────┤
  │  PHASE 4 — Continuous improvement                   │
  │  Rollback to best checkpoint if loss increases      │
  │  Hybrid trigger: federate on gradient drift OR      │
  │  every 6 hours                                      │
  └─────────────────────────────────────────────────────┘

Requirements (Raspberry Pi):
  pip install tflite-runtime paho-mqtt numpy python-can
"""

import os
os.environ["TFLITE_DISABLE_XNNPACK"] = "1"   # must be set before tflite import
import json
import time
import struct
import pickle
import queue
import threading
import numpy as np
from collections import deque
from datetime import datetime
import paho.mqtt.client as mqtt

# TFLite runtime — must support SELECT_TF_OPS (tflite-runtime >= 2.9)
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_OK = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_OK = True
    except ImportError:
        print("FATAL: tflite-runtime not found. "
              "Install: pip install tflite-runtime")
        TFLITE_OK = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_DIM            = 160        # 20 timesteps × 8 CAN features
WINDOW_SIZE          = 20
N_FEATURES           = 8
LOCAL_TRAIN_INTERVAL = 300        # seconds between local training cycles
MIN_BUFFER_FRAMES    = 200        # minimum CAN frames before first training
LOCAL_EPOCHS         = 3          # epochs per local training cycle
BATCH_SIZE           = 32
ANOMALY_PERCENTILE   = 97         # threshold calibration
ROLLBACK_PATIENCE    = 3          # rounds of worsening before rollback
FED_BASE_INTERVAL    = 21600      # 6 hours scheduled federation
FED_MIN_INTERVAL     = 1800       # 30 min minimum between rounds
DIVERGENCE_THRESHOLD = 0.10       # gradient norm drift trigger
_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR      = os.path.join(_BASE_DIR, "models")
# /tmp is always writable — tf.raw_ops.Save writes directly via OS,
# bypassing Python's makedirs, so it needs a path that already exists
CHECKPOINT_DIR = "/tmp/fed_ids_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# SECTION 1: On-device TFLite model wrapper
# ─────────────────────────────────────────────

class OnDeviceModel:
    """
    Stateless TFLite autoencoder.
    Weights are numpy arrays in Python, passed into every train/infer call.
    No resource variables → no segfaults on Pi tflite-runtime.
    """

    DIMS = [160, 64, 32, 16, 32, 64, 160]

    def __init__(self):
        self.interpreter  = None
        self.train_runner = None
        self.infer_runner = None
        self.threshold    = 0.05
        self.is_loaded    = False
        self.weights      = []   # [w0..w5]
        self.biases       = []   # [b0..b5]
        self._init_weights()

    def _init_weights(self):
        dims = self.DIMS
        self.weights, self.biases = [], []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(
                (np.random.randn(dims[i], dims[i+1]) * scale).astype(np.float32))
            self.biases.append(np.zeros(dims[i+1], dtype=np.float32))

    def load_from_bytes(self, tflite_bytes: bytes, dims: list = None) -> bool:
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "current_model.tflite")
        with open(path, "wb") as f:
            f.write(tflite_bytes)
        if dims:
            self.DIMS = dims
            self._init_weights()
        return self.load_from_file(path)

    def load_from_file(self, path: str) -> bool:
        if not TFLITE_OK or not os.path.exists(path):
            return False
        try:
            self.interpreter  = tflite.Interpreter(
                model_path=path,
                num_threads=1,
            )
            self.interpreter.allocate_tensors()
            self.train_runner = self.interpreter.get_signature_runner('train')
            self.infer_runner = self.interpreter.get_signature_runner('infer')
            self.is_loaded    = True
            print(f"[Model] Loaded: {path}")
            return True
        except Exception as e:
            print(f"[Model] Load error: {e}")
            return False

    def _weight_kwargs(self):
        kw = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            kw[f'w{i}'] = w
            kw[f'b{i}'] = b
        return kw

    # ── Training ─────────────────────────────────────────────────────────────

    def train_step(self, x_batch: np.ndarray) -> float:
        """One SGD step via TFLite train signature."""
        if not self.is_loaded:
            return float('inf')
        kw     = self._weight_kwargs()
        result = self.train_runner(x=x_batch.astype(np.float32), **kw)
        for i in range(len(self.weights)):
            self.weights[i] = result[f'w{i}'].copy()
            self.biases[i]  = result[f'b{i}'].copy()
        return float(result['loss'])

    def reload_interpreter(self):
        """
        Reload the TFLite interpreter from disk to clear accumulated
        memory state. Call between training epochs on tflite-runtime 2.14
        to prevent double-free corruption from repeated kwargs calls.
        """
        path = os.path.join(MODEL_DIR, "current_model.tflite")
        if os.path.exists(path):
            self.interpreter  = tflite.Interpreter(
                model_path=path,
                num_threads=1,
            )
            self.interpreter.allocate_tensors()
            self.train_runner = self.interpreter.get_signature_runner('train')
            self.infer_runner = self.interpreter.get_signature_runner('infer')

    # ── Inference (pure numpy — avoids repeated TFLite kwargs memory bug) ────

    def _numpy_forward(self, x: np.ndarray) -> np.ndarray:
        """Pure numpy forward pass using self.weights/biases."""
        h = x.reshape(1, INPUT_DIM).astype(np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            h = 1.0 / (1.0 + np.exp(-z)) if i < len(self.weights) - 1 else z
        return h

    def infer(self, x: np.ndarray):
        """Anomaly detection using pure numpy — no TFLite call."""
        if not self.weights:
            return False, 0.0
        output = self._numpy_forward(x)
        x_flat = x.reshape(1, INPUT_DIM).astype(np.float32)
        error  = float(np.mean((x_flat - output) ** 2))
        return error > self.threshold, error

    # ── Numpy checkpointing ───────────────────────────────────────────────────

    def save_checkpoint(self, name: str = "best") -> str:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, f"{name}.npz")
        np.savez(path, *self.weights, *self.biases)
        print(f"[Model] Checkpoint saved: {path}")
        return path

    def restore_checkpoint(self, name: str = "best"):
        path = os.path.join(CHECKPOINT_DIR, f"{name}.npz")
        if not os.path.exists(path):
            print(f"[Model] No checkpoint at {path}")
            return
        data   = np.load(path)
        arrays = [data[k] for k in sorted(data.files)]
        n = len(self.weights)
        self.weights = [arrays[i].astype(np.float32) for i in range(n)]
        self.biases  = [arrays[i+n].astype(np.float32) for i in range(n)]
        print(f"[Model] Restored: {path}")

    # ── FedAvg weight interface ───────────────────────────────────────────────

    def get_weights(self) -> list:
        """Return [w0,b0,w1,b1,...] for FedAvg. No TFLite call needed."""
        result = []
        for w, b in zip(self.weights, self.biases):
            result.extend([w.copy(), b.copy()])
        return result

    def set_weights_from_fedavg(self, flat_list: list):
        n = len(self.weights)
        for i in range(n):
            self.weights[i] = flat_list[i * 2].astype(np.float32)
            self.biases[i]  = flat_list[i * 2 + 1].astype(np.float32)

    # ── Threshold calibration ─────────────────────────────────────────────────

    def calibrate_threshold(self, normal_errors: list):
        if len(normal_errors) > 20:
            self.threshold = float(
                np.percentile(normal_errors, ANOMALY_PERCENTILE))
            print(f"[Model] Threshold: {self.threshold:.6f}")


# ─────────────────────────────────────────────
# SECTION 2: Rollback manager
# ─────────────────────────────────────────────

class RollbackManager:
    """
    Tracks loss history. If loss worsens for ROLLBACK_PATIENCE consecutive
    local training cycles, restores the best checkpoint automatically.
    """

    def __init__(self, model: OnDeviceModel):
        self.model        = model
        self.best_loss    = float('inf')
        self.worse_count  = 0
        self.loss_history = []

    def update(self, loss: float, round_label: str) -> bool:
        """
        Call after each local training cycle.
        Returns True if rollback was triggered.
        """
        self.loss_history.append(loss)

        if loss < self.best_loss:
            self.best_loss   = loss
            self.worse_count = 0
            self.model.save_checkpoint("best")
            print(f"[Rollback] Best checkpoint updated: loss={loss:.6f}")
            return False
        else:
            self.worse_count += 1
            print(f"[Rollback] Loss worsening "
                  f"({self.worse_count}/{ROLLBACK_PATIENCE}): "
                  f"current={loss:.6f}, best={self.best_loss:.6f}")
            if self.worse_count >= ROLLBACK_PATIENCE:
                self.model.restore_checkpoint("best")
                self.worse_count = 0
                print("[Rollback] ✓ Rolled back to best checkpoint")
                return True
        return False

    def is_improving(self) -> bool:
        """Return True if recent trend is improving (safe to federate)."""
        if len(self.loss_history) < 3:
            return True
        recent = self.loss_history[-3:]
        return recent[-1] <= recent[0]


# ─────────────────────────────────────────────
# SECTION 3: Federation trigger
# ─────────────────────────────────────────────

class FederationTrigger:
    """
    Decides when to send a weight update to the server.
    Three conditions:
      1. Gradient divergence > threshold  → immediate (early trigger)
      2. Loss spike > 30%                 → immediate (concept drift)
      3. 6 hours elapsed                  → scheduled (skip if stable)
    """

    def __init__(self):
        self.ref_weights    = None
        self.ref_loss       = None
        self.last_fed_time  = time.time()
        self.adaptive_interval = FED_BASE_INTERVAL

    def store_reference(self, weights: list, loss: float):
        self.ref_weights   = [w.copy() for w in weights]
        self.ref_loss      = loss
        self.last_fed_time = time.time()

    def _divergence(self, current_weights: list) -> float:
        if self.ref_weights is None or len(self.ref_weights) == 0:
            return float('inf')
        total_delta, total_n = 0.0, 0
        for curr, ref in zip(current_weights, self.ref_weights):
            total_delta += float(np.sum((curr - ref) ** 2))
            total_n     += curr.size
        return float(np.sqrt(total_delta / max(total_n, 1)))

    def should_federate(self, current_weights: list,
                        current_loss: float):
        """
        Returns (should_federate: bool, reason: str)
        """
        elapsed    = time.time() - self.last_fed_time
        divergence = self._divergence(current_weights)
        loss_change = (
            abs(current_loss - self.ref_loss) / max(self.ref_loss, 1e-8)
            if self.ref_loss is not None else float('inf')
        )

        # Early trigger: significant gradient drift
        if (divergence > DIVERGENCE_THRESHOLD and
                elapsed > FED_MIN_INTERVAL):
            return True, f"gradient_divergence={divergence:.4f}"

        # Early trigger: concept drift (anomaly / new attack pattern)
        if loss_change > 0.30 and elapsed > FED_MIN_INTERVAL:
            self.adaptive_interval = FED_BASE_INTERVAL  # reset
            return True, f"concept_drift={loss_change:.2%}"

        # Scheduled trigger
        if elapsed >= self.adaptive_interval:
            if divergence > 0.01:  # something meaningful to share
                return True, "scheduled"
            else:
                # Model is stable — extend interval, skip round
                self.adaptive_interval = min(
                    self.adaptive_interval * 1.5,
                    FED_BASE_INTERVAL * 3)
                self.last_fed_time = time.time()
                return False, "skipped_stable"

        return False, f"waiting {elapsed/3600:.1f}h/{self.adaptive_interval/3600:.1f}h"


# ─────────────────────────────────────────────
# SECTION 4: CAN data collector
# ─────────────────────────────────────────────

class DataCollector:
    """
    Reads from CAN bus (or simulates data if unavailable).
    Maintains a rolling buffer and provides normalised sliding windows.
    """

    FEATURE_NAMES = [
        'speed_kmh', 'battery_level', 'throttle', 'brake',
        'steering', 'gear', 'location_x', 'location_y',
    ]

    def __init__(self, can_interface: str = 'can0',
                 buffer_size: int = 2000):
        self.can_interface = can_interface
        self.buffer        = deque(maxlen=buffer_size)
        self.state         = {n: 0.0 for n in self.FEATURE_NAMES}
        self.running       = False

        # Normalisation stats (fitted after MIN_BUFFER_FRAMES collected)
        self.mean          = np.zeros(N_FEATURES, dtype=np.float32)
        self.std           = np.ones(N_FEATURES,  dtype=np.float32)
        self.scaler_fitted = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self._can_listener, daemon=True)
        t.start()
        print(f"[DataCollector] Started on {self.can_interface}")

    def stop(self):
        self.running = False

    def _can_listener(self):
        try:
            import can
            bus = can.Bus(channel=self.can_interface,
                          interface='socketcan')
            while self.running:
                msg = bus.recv(timeout=1.0)
                if msg:
                    self._process(msg)
        except Exception as e:
            print(f"[DataCollector] CAN error: {e} — using simulated data")
            self._simulate()

    def _process(self, msg):
        cid = msg.arbitration_id
        try:
            v = struct.unpack('<f', msg.data[:4])[0]
            m = {0x123: 'speed_kmh',    0x124: 'battery_level',
                 0x125: 'throttle',     0x126: 'brake',
                 0x127: 'steering',     0x128: 'gear'}
            if cid in m:
                self.state[m[cid]] = v
            self._push()
        except Exception:
            pass

    def _simulate(self):
        import random
        while self.running:
            s = self.state
            s['speed_kmh']     = max(0, s.get('speed_kmh',50) + random.gauss(0,2))
            s['battery_level'] = max(0, min(100, s.get('battery_level',80) - 0.01))
            s['throttle']      = max(0, min(1, 0.3 + random.gauss(0, 0.1)))
            s['brake']         = max(0, min(1, abs(random.gauss(0, 0.05))))
            s['steering']      = random.gauss(0, 0.1)
            s['gear']          = float(random.randint(0, 5))
            s['location_x']    = s.get('location_x', 0) + random.gauss(0, 0.5)
            s['location_y']    = s.get('location_y', 0) + random.gauss(0, 0.5)
            self._push()
            time.sleep(0.1)

    def _push(self):
        row = np.array([self.state.get(n, 0.0)
                        for n in self.FEATURE_NAMES], dtype=np.float32)
        self.buffer.append(row)

    def fit_scaler(self):
        if len(self.buffer) < 50:
            return
        data      = np.array(list(self.buffer))
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std  = np.where(data.std(axis=0) > 1e-6,
                             data.std(axis=0), 1.0).astype(np.float32)
        self.scaler_fitted = True

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def ready(self) -> bool:
        return len(self.buffer) >= MIN_BUFFER_FRAMES

    def get_windows(self, stride: int = 5,
                    max_windows: int = 400) -> np.ndarray:
        """Return (N, 160) normalised sliding-window array for training."""
        if not self.ready():
            return None
        if not self.scaler_fitted:
            self.fit_scaler()
        data = self._normalise(
            np.array(list(self.buffer), dtype=np.float32))
        out  = []
        for i in range(0, min(len(data) - WINDOW_SIZE,
                              max_windows * stride), stride):
            out.append(data[i:i + WINDOW_SIZE].flatten())
        return np.array(out, dtype=np.float32) if out else None

    def get_latest_window(self) -> np.ndarray:
        """Return (160,) most recent window for real-time inference."""
        if len(self.buffer) < WINDOW_SIZE:
            return None
        if not self.scaler_fitted:
            self.fit_scaler()
        data = self._normalise(
            np.array(list(self.buffer)[-WINDOW_SIZE:], dtype=np.float32))
        return data.flatten()


# ─────────────────────────────────────────────
# SECTION 5: Main federated client
# ─────────────────────────────────────────────

class FederatedClient:
    """
    Orchestrates the full federated learning lifecycle on the Pi.
    """

    def __init__(self, client_id: str, broker: str,
                 port: int = 1883, can_interface: str = 'can0'):
        self.client_id = client_id
        self.broker    = broker
        self.port      = port

        # Core components
        self.model     = OnDeviceModel()
        self.collector = DataCollector(can_interface)
        self.rollback  = RollbackManager(self.model)
        self.trigger   = FederationTrigger()

        # MQTT
        self.mqtt         = None
        self.connected    = False

        # State
        self.current_round    = 0
        self.model_queue      = queue.Queue()
        self.running          = False
        self.last_train_time  = 0.0
        self.current_loss     = float('inf')
        self.normal_errors    = deque(maxlen=500)
        self.phase            = "waiting_for_model"  # lifecycle phase

        # Load any previously saved model from disk
        self._try_load_saved_model()

    def _try_load_saved_model(self):
        path = os.path.join(MODEL_DIR, "current_model.tflite")
        if os.path.exists(path):
            if self.model.load_from_file(path):
                self.phase = "local_pretraining"
                print("[Client] Loaded saved model from disk")

    # ── MQTT ──────────────────────────────────────────────────────────────────

    def connect(self):
        try:
            self.mqtt = mqtt.Client(
                client_id=f"fed_{self.client_id}",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        except Exception:
            self.mqtt = mqtt.Client(client_id=f"fed_{self.client_id}")

        self.mqtt.on_connect    = self._on_connect
        self.mqtt.on_message    = self._on_message
        self.mqtt.on_disconnect = self._on_disconnect

        print(f"[Client] Connecting to {self.broker}:{self.port}...")
        self.mqtt.connect(self.broker, self.port, keepalive=60)
        self.mqtt.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if hasattr(rc, 'value'):
            rc = rc.value
        if rc == 0:
            self.connected = True
            print(f"[Client {self.client_id}] Connected to MQTT broker")
            client.subscribe("federated/model/global")
            client.subscribe("federated/aggregation/trigger")
            client.subscribe(f"federated/clients/{self.client_id}/commands")
            self._publish_status("online")
        else:
            print(f"[Client] MQTT connect failed: rc={rc}")

    def _on_disconnect(self, client, userdata, disconnect_flags,
                       rc, properties=None):
        self.connected = False
        print(f"[Client {self.client_id}] Disconnected — will reconnect")

    def _on_message(self, client, userdata, message):
        topic = message.topic
        try:
            if topic == "federated/model/global":
                data = pickle.loads(message.payload)
                self.model_queue.put(data)

            elif topic == "federated/aggregation/trigger":
                print(f"[Client {self.client_id}] Server requested training round")
                threading.Thread(target=self._training_cycle,
                                 daemon=True).start()
        except Exception as e:
            print(f"[Client] Message error: {e}")

    def _publish_status(self, status: str):
        if not self.connected:
            return
        self.mqtt.publish("federated/clients/status", json.dumps({
            "client_id": self.client_id,
            "status":    status,
            "phase":     self.phase,
            "round":     self.current_round,
            "loss":      self.current_loss,
            "timestamp": time.time(),
        }))

    # ── Apply global model ────────────────────────────────────────────────────

    def _apply_global_model(self, data: dict):
        """Load new .tflite from server and update weights."""
        round_num    = data.get("round", self.current_round + 1)
        tflite_bytes = data.get("tflite_bytes")
        dims         = data.get("dims", None)

        if tflite_bytes is None:
            print("[Client] Global model payload missing tflite_bytes")
            return

        print(f"[Client] Applying global model from round {round_num}...")

        # If server sent averaged weights (FedAvg round), apply them first
        fedavg_weights = data.get("fedavg_weights")
        if fedavg_weights is not None:
            self.model.set_weights_from_fedavg(fedavg_weights)

        if self.model.load_from_bytes(tflite_bytes, dims):
            self.current_round = round_num
            self.phase         = "local_pretraining"
            weights = self.model.get_weights()
            if weights:
                self.trigger.store_reference(weights, self.current_loss)
            print(f"[Client] Global model applied ✓ (round {round_num})")
        else:
            print("[Client] Failed to load global model")

    # ── Training cycle ────────────────────────────────────────────────────────

    def _training_cycle(self):
        """
        One full local training cycle:
          1. Get windows from CAN buffer
          2. Run LOCAL_EPOCHS of mini-batch training
          3. Rollback check
          4. Federation trigger check → send weights if triggered
          5. Calibrate anomaly threshold
        """
        if not self.model.is_loaded:
            print("[Client] No model loaded yet — skipping training")
            return
        if not self.collector.ready():
            print(f"[Client] Buffer not ready "
                  f"({len(self.collector.buffer)}/{MIN_BUFFER_FRAMES} frames)")
            return

        self._publish_status("training")

        windows = self.collector.get_windows(stride=5)
        if windows is None:
            self._publish_status("idle")
            return

        # ── Local training ────────────────────────────────────────────────
        losses = []
        for epoch in range(LOCAL_EPOCHS):
            # Reload interpreter between epochs to clear tflite-runtime
            # 2.14 memory state — prevents double-free corruption
            if epoch > 0:
                self.model.reload_interpreter()
            idx  = np.random.permutation(len(windows))
            for i in range(0, len(windows), BATCH_SIZE):
                batch = windows[idx[i:i + BATCH_SIZE]]
                if len(batch) == 0:
                    continue
                loss = self.model.train_step(batch)
                losses.append(loss)
        # Reload once more after training to reset interpreter state
        self.model.reload_interpreter()

        avg_loss = float(np.mean(losses)) if losses else float('inf')
        self.current_loss = avg_loss
        print(f"[Client {self.client_id}] "
              f"Training: loss={avg_loss:.6f}, "
              f"windows={len(windows)}, phase={self.phase}")

        # ── Rollback check ────────────────────────────────────────────────
        rolled_back = self.rollback.update(avg_loss, f"round_{self.current_round}")
        if rolled_back:
            self._publish_status("rolled_back")
            # Do NOT send weights after rollback — weights are reverted
            return

        # ── Federation trigger ────────────────────────────────────────────
        current_weights = self.model.get_weights()
        should_fed, reason = self.trigger.should_federate(
            current_weights, avg_loss)

        if should_fed and self.rollback.is_improving():
            self._send_weight_update(current_weights, avg_loss, len(windows))
            self.trigger.store_reference(current_weights, avg_loss)
        else:
            print(f"[Client] Federation: {reason}")

        # ── Threshold calibration ─────────────────────────────────────────
        sample_windows = windows[:100]
        for w in sample_windows:
            _, err = self.model.infer(w)
            self.normal_errors.append(err)
        self.model.calibrate_threshold(list(self.normal_errors))

        self._publish_status("idle")
        self.last_train_time = time.time()

    def _send_weight_update(self, weights: list, loss: float,
                             num_samples: int):
        """
        Send weight arrays (NOT raw data) to server for FedAvg.
        This is the only thing the server ever receives from the Pi.
        """
        if not self.connected:
            print("[Client] Not connected — weight update queued for next connection")
            return

        payload = pickle.dumps({
            "client_id":   self.client_id,
            "round":       self.current_round,
            "weights":     weights,          # list of np.ndarray
            "num_samples": num_samples,
            "loss":        loss,
            "timestamp":   time.time(),
        })
        self.mqtt.publish("federated/model/updates", payload, qos=1)
        print(f"[Client {self.client_id}] "
              f"Sent weight update: "
              f"round={self.current_round}, "
              f"loss={loss:.6f}, "
              f"samples={num_samples}, "
              f"size={len(payload)/1024:.1f} KB")

    # ── Anomaly detection (real-time, every 100ms) ────────────────────────────

    def _detect_anomaly(self):
        window = self.collector.get_latest_window()
        if window is None or not self.model.is_loaded:
            return
        is_anomaly, score = self.model.infer(window)
        if is_anomaly:
            print(f"[Client {self.client_id}] ⚠️  ANOMALY: score={score:.6f}")
            self.mqtt.publish("federated/alerts/attack", json.dumps({
                "client_id": self.client_id,
                "score":     score,
                "threshold": self.model.threshold,
                "round":     self.current_round,
                "timestamp": time.time(),
            }))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print(f"FEDERATED IDS CLIENT — {self.client_id}")
        print(f"  Broker        : {self.broker}:{self.port}")
        print(f"  CAN interface : {self.collector.can_interface}")
        print(f"  Fed interval  : {FED_BASE_INTERVAL/3600:.0f}h (base)")
        print("=" * 60)

        self.collector.start()
        self.connect()
        time.sleep(2)
        self.running = True

        inference_tick = 0

        try:
            while self.running:

                # ── Apply any received global models ──────────────────────
                try:
                    model_data = self.model_queue.get_nowait()
                    self._apply_global_model(model_data)
                except queue.Empty:
                    pass

                # ── Periodic local training ───────────────────────────────
                now = time.time()
                if (now - self.last_train_time >= LOCAL_TRAIN_INTERVAL and
                        self.model.is_loaded and
                        self.collector.ready()):
                    threading.Thread(target=self._training_cycle,
                                     daemon=True).start()

                # ── Real-time anomaly detection (every 100ms) ─────────────
                self._detect_anomaly()

                # ── Phase logging (every ~30s) ────────────────────────────
                inference_tick += 1
                if inference_tick % 300 == 0:
                    print(f"[Client {self.client_id}] "
                          f"phase={self.phase} | "
                          f"round={self.current_round} | "
                          f"loss={self.current_loss:.6f} | "
                          f"buffer={len(self.collector.buffer)}")

                time.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\n[Client {self.client_id}] Shutting down...")
        finally:
            # Save model on exit so it survives reboot
            self.model.save_checkpoint("shutdown")
            self.collector.stop()
            if self.mqtt:
                self.mqtt.loop_stop()
                try:
                    self.mqtt.disconnect()
                except Exception:
                    pass


def main():
    import argparse
    p = argparse.ArgumentParser(description="Federated IDS Edge Client")
    p.add_argument("--client-id",      default="edge_1",
                   help="Unique ID for this Pi (e.g. edge_1, edge_2, edge_3)")
    p.add_argument("--broker",         required=True,
                   help="MQTT broker IP (Windows server IP)")
    p.add_argument("--port",           type=int, default=1883)
    p.add_argument("--can-interface",  default="can0")
    args = p.parse_args()

    client = FederatedClient(
        client_id=args.client_id,
        broker=args.broker,
        port=args.port,
        can_interface=args.can_interface,
    )
    client.run()


if __name__ == "__main__":
    main()
