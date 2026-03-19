# fed_server.py  –  Federated Learning Server (Windows)
"""
Central aggregation server. Runs on Windows alongside the MQTT broker.

Responsibilities:
  1. Wait for min_clients Pi devices to connect and send weight arrays
  2. Run FedAvg (weighted average of weight arrays by num_samples)
  3. Rebuild a new trainable TFLite model with the averaged weights
  4. Push the updated .tflite back to all Pi clients
  5. Repeat every federation round (triggered by client updates OR 6-hour schedule)

The server NEVER sees raw CAN data — only:
  - Weight arrays (list of numpy ndarrays)
  - num_samples (for weighted averaging)
  - loss (for monitoring)

Requirements (Windows server):
  pip install tensorflow paho-mqtt numpy
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
MQTT_BROKER        = "127.0.0.1"
MQTT_PORT          = 1883
MIN_CLIENTS        = 2           # Wait for at least this many updates per round
MAX_WAIT_SEC       = 21600       # 6 hours max wait before round with whatever arrived
MODEL_PATH         = "./models/can_autoencoder_trainable.tflite"
CHECKPOINT_DIR     = "./models/checkpoints"
INPUT_DIM          = 160
LEARNING_RATE      = 0.001
# ─────────────────────────────────────────────────────────────────────────────


class FedAvgAggregator:
    """
    Weighted FedAvg over lists of numpy weight arrays.
    Each client sends: {'weights': [np.ndarray, ...], 'num_samples': int}
    """

    def __init__(self):
        self.updates = {}       # client_id → {weights, num_samples, loss, round}
        self.lock    = threading.Lock()

    def add_update(self, client_id: str, weights: list,
                   num_samples: int, loss: float, round_num: int):
        with self.lock:
            self.updates[client_id] = {
                "weights":     weights,
                "num_samples": num_samples,
                "loss":        loss,
                "round":       round_num,
                "timestamp":   time.time(),
            }
            print(f"[FedAvg] Received update from {client_id}: "
                  f"samples={num_samples}, loss={loss:.6f}, round={round_num}")

    def n_updates(self) -> int:
        with self.lock:
            return len(self.updates)

    def aggregate(self) -> list:
        """
        Compute weighted average of all received weight arrays.
        Returns list of numpy ndarrays (same structure as client weights).
        """
        with self.lock:
            if not self.updates:
                return None

            total_samples = sum(u["num_samples"] for u in self.updates.values())
            if total_samples == 0:
                return None

            averaged = None
            for client_id, update in self.updates.items():
                weight = update["num_samples"] / total_samples
                params = update["weights"]

                if averaged is None:
                    averaged = [w.astype(np.float32) * weight for w in params]
                else:
                    for i, w in enumerate(params):
                        averaged[i] += w.astype(np.float32) * weight

            # Log round summary
            losses = [u["loss"] for u in self.updates.values()]
            print(f"[FedAvg] Aggregated {len(self.updates)} clients | "
                  f"total_samples={total_samples} | "
                  f"avg_loss={np.mean(losses):.6f} | "
                  f"min_loss={np.min(losses):.6f}")

            self.updates.clear()
            return averaged


class TFLiteRebuilder:
    """
    Takes averaged weight arrays from FedAvg and rebuilds a trainable
    .tflite model with those weights baked in, ready to push to Pis.
    """

    def __init__(self, model_path: str, checkpoint_dir: str):
        self.model_path     = model_path
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def rebuild_with_weights(self, averaged_weights: list,
                              round_num: int) -> bytes:
        """
        Load the base TFLite model architecture, apply averaged weights,
        re-export as a new trainable .tflite. Returns bytes.
        """
        import tensorflow as tf

        # We rebuild from the Keras model definition to apply new weights
        model = self._build_keras_model()

        # Apply averaged weights in order: kernel, bias, kernel, bias, ...
        trainable_vars = model.trainable_variables
        if len(trainable_vars) != len(averaged_weights):
            print(f"[Rebuilder] WARNING: weight count mismatch: "
                  f"model={len(trainable_vars)}, received={len(averaged_weights)}")
            # Try to apply as many as match
            n = min(len(trainable_vars), len(averaged_weights))
            for i in range(n):
                trainable_vars[i].assign(
                    averaged_weights[i].reshape(trainable_vars[i].shape))
        else:
            for var, w in zip(trainable_vars, averaged_weights):
                var.assign(w.reshape(var.shape))

        # Wrap in the trainable module and re-export
        tflite_bytes = self._export_to_tflite(model)

        # Save checkpoint
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self.checkpoint_dir,
                           f"global_round{round_num}_{ts}.tflite")
        with open(out, "wb") as f:
            f.write(tflite_bytes)
        print(f"[Rebuilder] New global model saved: {out} "
              f"({len(tflite_bytes)/1024:.1f} KB)")
        return tflite_bytes

    def _build_keras_model(self):
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(INPUT_DIM,)),
            tf.keras.layers.Dense(64,  activation='relu',
                                  kernel_initializer='he_normal'),
            tf.keras.layers.Dense(32,  activation='relu',
                                  kernel_initializer='he_normal'),
            tf.keras.layers.Dense(16,  activation='relu',
                                  kernel_initializer='he_normal'),
            tf.keras.layers.Dense(32,  activation='relu',
                                  kernel_initializer='he_normal'),
            tf.keras.layers.Dense(64,  activation='relu',
                                  kernel_initializer='he_normal'),
            tf.keras.layers.Dense(INPUT_DIM),
        ])
        return model

    def _export_to_tflite(self, keras_model) -> bytes:
        import tensorflow as tf

        class TrainableModule(tf.Module):
            def __init__(self, model):
                super().__init__()
                self.model   = model
                self._loss   = tf.keras.losses.MeanSquaredError()
                self._optim  = tf.keras.optimizers.Adam(
                    learning_rate=LEARNING_RATE)

            @tf.function(input_signature=[
                tf.TensorSpec([None, INPUT_DIM], tf.float32, name='x')])
            def train(self, x):
                with tf.GradientTape() as tape:
                    pred = self.model(x, training=True)
                    loss = self._loss(x, pred)
                grads = tape.gradient(loss, self.model.trainable_variables)
                self._optim.apply_gradients(
                    zip(grads, self.model.trainable_variables))
                return {"loss": loss}

            @tf.function(input_signature=[
                tf.TensorSpec([None, INPUT_DIM], tf.float32, name='x')])
            def infer(self, x):
                pred = self.model(x, training=False)
                mse  = tf.reduce_mean(tf.square(x - pred), axis=1)
                return {"output": pred, "reconstruction_error": mse}

            @tf.function(input_signature=[
                tf.TensorSpec(shape=[], dtype=tf.string,
                              name='checkpoint_path')])
            def save(self, checkpoint_path):
                names   = [w.name for w in self.model.weights]
                tensors = [w.read_value() for w in self.model.weights]
                tf.raw_ops.Save(filename=checkpoint_path,
                                tensor_names=names,
                                data=tensors, name='save')
                return {"checkpoint_path": checkpoint_path}

            @tf.function(input_signature=[
                tf.TensorSpec(shape=[], dtype=tf.string,
                              name='checkpoint_path')])
            def restore(self, checkpoint_path):
                for w in self.model.weights:
                    restored = tf.raw_ops.Restore(
                        file_pattern=checkpoint_path,
                        tensor_name=w.name,
                        dt=w.dtype, name='restore')
                    w.assign(restored)
                return {"checkpoint_path": checkpoint_path}

        module = TrainableModule(keras_model)
        dummy  = tf.zeros([1, INPUT_DIM])
        module.train(dummy)
        module.infer(dummy)

        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [
                module.train.get_concrete_function(),
                module.infer.get_concrete_function(),
                module.save.get_concrete_function(),
                module.restore.get_concrete_function(),
            ],
            module,
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.experimental_enable_resource_variables = True
        return converter.convert()


class FederatedServer:
    """Main server: receives weight updates, runs FedAvg, pushes new model."""

    def __init__(self, broker=MQTT_BROKER, port=MQTT_PORT,
                 min_clients=MIN_CLIENTS):
        self.broker      = broker
        self.port        = port
        self.min_clients = min_clients

        self.aggregator = FedAvgAggregator()
        self.rebuilder  = TFLiteRebuilder(MODEL_PATH, CHECKPOINT_DIR)

        self.mqtt_client  = None
        self.connected    = False
        self.current_round = 0
        self.round_start  = time.time()

        self.known_clients = set()
        self._round_lock   = threading.Lock()
        self._round_in_progress = False

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
        """Parse incoming weight update and add to FedAvg aggregator."""
        try:
            update = pickle.loads(payload)

            client_id   = update["client_id"]
            weights     = update["weights"]       # list of np.ndarray
            num_samples = update["num_samples"]
            loss        = update["loss"]
            round_num   = update.get("round", self.current_round)

            self.aggregator.add_update(
                client_id, weights, num_samples, loss, round_num)

            n = self.aggregator.n_updates()
            print(f"[Server] Updates collected: {n}/{self.min_clients}")

            # Trigger aggregation if enough clients have responded
            if n >= self.min_clients and not self._round_in_progress:
                threading.Thread(target=self._run_fedavg_round,
                                 daemon=True).start()

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
            print(f"{'='*50}")

            # Aggregate weights
            averaged_weights = self.aggregator.aggregate()
            if averaged_weights is None:
                print("[Server] No updates to aggregate")
                return

            # Rebuild TFLite with averaged weights
            tflite_bytes = self.rebuilder.rebuild_with_weights(
                averaged_weights, self.current_round)

            # Broadcast new global model to all clients
            self._broadcast_global_model(tflite_bytes)

            self.round_start = time.time()
            print(f"[Server] Round {self.current_round} complete ✓\n")

        except Exception as e:
            print(f"[Server] FedAvg round error: {e}")
        finally:
            self._round_in_progress = False

    def _broadcast_global_model(self, tflite_bytes: bytes):
        """Publish updated global model to all Pi clients."""
        payload = pickle.dumps({
            "type":         "global_model",
            "round":        self.current_round,
            "tflite_bytes": tflite_bytes,
            "input_dim":    INPUT_DIM,
            "timestamp":    time.time(),
        })
        result = self.mqtt_client.publish(
            "federated/model/global", payload, qos=1, retain=True)
        result.wait_for_publish()
        print(f"[Server] Global model broadcast: "
              f"round={self.current_round}, "
              f"size={len(payload)/1024:.1f} KB")

    # ── Scheduled fallback round ───────────────────────────────────────────────

    def _scheduled_round_watcher(self):
        """
        If MAX_WAIT_SEC elapses without enough clients, run anyway
        with whatever updates arrived (at least 1).
        """
        while True:
            time.sleep(60)  # check every minute
            elapsed = time.time() - self.round_start
            n       = self.aggregator.n_updates()

            if elapsed >= MAX_WAIT_SEC and n >= 1 and not self._round_in_progress:
                print(f"[Server] Scheduled round triggered after "
                      f"{elapsed/3600:.1f}h with {n} update(s)")
                threading.Thread(target=self._run_fedavg_round,
                                 daemon=True).start()

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("FEDERATED LEARNING IDS SERVER")
        print(f"  Min clients per round : {self.min_clients}")
        print(f"  Max wait per round    : {MAX_WAIT_SEC/3600:.0f} hours")
        print(f"  MQTT broker           : {self.broker}:{self.port}")
        print("=" * 60)

        self.start_mqtt()

        # Start scheduled fallback watcher
        threading.Thread(target=self._scheduled_round_watcher,
                         daemon=True).start()

        # Wait for connection
        for _ in range(20):
            if self.connected:
                break
            time.sleep(0.5)

        if not self.connected:
            print("ERROR: Could not connect to MQTT broker. "
                  "Is Mosquitto running?")
            return

        print(f"\nServer ready. Waiting for {self.min_clients} client updates...")

        try:
            while True:
                n = self.aggregator.n_updates()
                elapsed_h = (time.time() - self.round_start) / 3600
                print(f"  [Round {self.current_round}] "
                      f"Updates: {n}/{self.min_clients} | "
                      f"Clients seen: {len(self.known_clients)} | "
                      f"Elapsed: {elapsed_h:.1f}h",
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
    parser = argparse.ArgumentParser(description="Federated IDS Server")
    parser.add_argument("--broker",      default=MQTT_BROKER)
    parser.add_argument("--port",        type=int, default=MQTT_PORT)
    parser.add_argument("--min-clients", type=int, default=MIN_CLIENTS)
    args = parser.parse_args()

    server = FederatedServer(
        broker=args.broker,
        port=args.port,
        min_clients=args.min_clients,
    )
    server.run()


if __name__ == "__main__":
    main()
