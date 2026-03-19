# model_builder.py  –  Run ONCE on the Windows server
"""
Creates a trainable TFLite model and publishes it to all Pi clients via MQTT.

The server NEVER sees raw CAN data. It only:
  1. Creates the initial untrained model architecture
  2. Converts it to a trainable .tflite with train/infer/save/restore signatures
  3. Pushes the .tflite bytes to all Pi clients so they can begin local training
  4. Later (fed_server.py) receives weight arrays from Pis, runs FedAvg, pushes back

Run order:
  Step 1:  python model_builder.py          ← this file (once)
  Step 2:  python fed_server.py             ← keeps running
  Step 3:  python fed_client.py on each Pi  ← keeps running

Requirements (Windows server only):
  pip install tensorflow paho-mqtt numpy
"""

import os
import json
import time
import pickle
import numpy as np
import paho.mqtt.client as mqtt

# ── CONFIG ────────────────────────────────────────────────────────────────────
MQTT_BROKER   = "127.0.0.1"
MQTT_PORT     = 1883
MODEL_SAVE    = "./models/can_autoencoder_trainable.tflite"
INPUT_DIM     = 160   # 20 timesteps × 8 CAN features
LEARNING_RATE = 0.001
# ─────────────────────────────────────────────────────────────────────────────


def build_trainable_tflite():
    """
    Build a CAN-bus autoencoder with four exported TFLite signatures:
      - 'train'   : one gradient step, returns loss
      - 'infer'   : reconstruction + per-sample MSE
      - 'save'    : save weights to checkpoint path
      - 'restore' : restore weights from checkpoint path
    No raw data ever leaves the Pi — only weight arrays are shared.
    """
    import tensorflow as tf

    class CANAutoencoder(tf.Module):
        def __init__(self):
            super().__init__()
            # Architecture: 160 → 64 → 32 → 16 → 32 → 64 → 160
            self.model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(INPUT_DIM,)),
                tf.keras.layers.Dense(64,  activation='relu',
                                      kernel_initializer='he_normal'),
                tf.keras.layers.Dense(32,  activation='relu',
                                      kernel_initializer='he_normal'),
                tf.keras.layers.Dense(16,  activation='relu',
                                      kernel_initializer='he_normal'),  # bottleneck
                tf.keras.layers.Dense(32,  activation='relu',
                                      kernel_initializer='he_normal'),
                tf.keras.layers.Dense(64,  activation='relu',
                                      kernel_initializer='he_normal'),
                tf.keras.layers.Dense(INPUT_DIM),  # linear output
            ])
            self._loss_fn = tf.keras.losses.MeanSquaredError()
            self._optim   = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        # ── TRAIN signature ──────────────────────────────────────────────────
        @tf.function(input_signature=[
            tf.TensorSpec([None, INPUT_DIM], tf.float32, name='x'),
        ])
        def train(self, x):
            """One Adam step. Autoencoder: target == input."""
            with tf.GradientTape() as tape:
                prediction = self.model(x, training=True)
                loss = self._loss_fn(x, prediction)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self._optim.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            return {"loss": loss}

        # ── INFER signature ──────────────────────────────────────────────────
        @tf.function(input_signature=[
            tf.TensorSpec([None, INPUT_DIM], tf.float32, name='x'),
        ])
        def infer(self, x):
            """Reconstruction + per-sample MSE anomaly score."""
            prediction = self.model(x, training=False)
            mse = tf.reduce_mean(tf.square(x - prediction), axis=1)
            return {
                "output":               prediction,
                "reconstruction_error": mse,
            }

        # ── SAVE signature ───────────────────────────────────────────────────
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[], dtype=tf.string, name='checkpoint_path'),
        ])
        def save(self, checkpoint_path):
            """Save model weights to a TF checkpoint on-device."""
            tensor_names    = [w.name for w in self.model.weights]
            tensors_to_save = [w.read_value() for w in self.model.weights]
            tf.raw_ops.Save(
                filename=checkpoint_path,
                tensor_names=tensor_names,
                data=tensors_to_save,
                name='save')
            return {"checkpoint_path": checkpoint_path}

        # ── RESTORE signature ────────────────────────────────────────────────
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[], dtype=tf.string, name='checkpoint_path'),
        ])
        def restore(self, checkpoint_path):
            """Restore model weights from a TF checkpoint."""
            for w in self.model.weights:
                restored = tf.raw_ops.Restore(
                    file_pattern=checkpoint_path,
                    tensor_name=w.name,
                    dt=w.dtype,
                    name='restore')
                w.assign(restored)
            return {"checkpoint_path": checkpoint_path}

    # ── Convert to TFLite ────────────────────────────────────────────────────
    module = CANAutoencoder()

    # Warm up so variables are created before conversion
    dummy = tf.zeros([1, INPUT_DIM])
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
        tf.lite.OpsSet.SELECT_TF_OPS,  # needed for Save/Restore raw ops
    ]
    converter.experimental_enable_resource_variables = True

    tflite_bytes = converter.convert()
    os.makedirs("./models", exist_ok=True)
    with open(MODEL_SAVE, "wb") as f:
        f.write(tflite_bytes)

    size_kb = len(tflite_bytes) / 1024
    print(f"✓ Trainable TFLite model built: {MODEL_SAVE} ({size_kb:.1f} KB)")
    return tflite_bytes


def push_model_to_clients(tflite_bytes: bytes):
    """Publish the trainable .tflite to all Pi clients via MQTT."""
    connected = {"ok": False}

    def on_connect(client, userdata, flags, rc, properties=None):
        if hasattr(rc, 'value'):
            rc = rc.value
        if rc == 0:
            connected["ok"] = True

    client = mqtt.Client(client_id="model_builder")
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()

    # Wait for connection
    for _ in range(20):
        if connected["ok"]:
            break
        time.sleep(0.5)

    if not connected["ok"]:
        print("ERROR: Could not connect to MQTT broker")
        return

    payload = pickle.dumps({
        "type":         "initial_model",
        "round":        0,
        "tflite_bytes": tflite_bytes,
        "input_dim":    INPUT_DIM,
        "timestamp":    time.time(),
    })

    # Publish with retain=True so late-joining Pis still receive it
    result = client.publish("federated/model/global", payload,
                            qos=1, retain=True)
    result.wait_for_publish()
    print(f"✓ Model pushed to clients ({len(payload)/1024:.1f} KB) "
          f"on federated/model/global")

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    print("=" * 60)
    print("CAN-BUS IDS — MODEL BUILDER")
    print("Builds trainable TFLite model and pushes to Pi clients")
    print("=" * 60)

    # Check if model already exists
    if os.path.exists(MODEL_SAVE):
        print(f"Found existing model: {MODEL_SAVE}")
        ans = input("Rebuild? [y/N]: ").strip().lower()
        if ans == 'y':
            tflite_bytes = build_trainable_tflite()
        else:
            with open(MODEL_SAVE, "rb") as f:
                tflite_bytes = f.read()
            print(f"Loaded existing model ({len(tflite_bytes)/1024:.1f} KB)")
    else:
        tflite_bytes = build_trainable_tflite()

    push_model_to_clients(tflite_bytes)
    print("\nDone. Now start fed_server.py and then fed_client.py on each Pi.")
