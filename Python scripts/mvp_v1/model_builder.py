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
    Checkpointing is handled in Python via numpy (no Flex delegate needed).
    No raw data ever leaves the Pi — only weight arrays are shared.
    """
    import tensorflow as tf

    # ── Manual autoencoder using only TFLite-native ops ──────────────────────
    # GradientTape + Keras relu generates ReluGrad/BroadcastGradientArgs
    # which are NOT native TFLite ops. Manual backprop avoids this entirely.
    # Architecture: 160 → 64 → 32 → 16 → 32 → 64 → 160
    # Activation: sigmoid (differentiable with native TFLite ops)
    # Adam optimizer implemented manually with tfl.var_handle ops

    DIMS = [INPUT_DIM, 64, 32, 16, 32, 64, INPUT_DIM]

    class CANAutoencoder(tf.Module):
        def __init__(self):
            super().__init__()
            self.lr    = tf.Variable(LEARNING_RATE, trainable=False,
                                     dtype=tf.float32, name='lr')
            self.step  = tf.Variable(0, trainable=False,
                                     dtype=tf.int64, name='step')
            self.b1    = tf.constant(0.9,   dtype=tf.float32)
            self.b2    = tf.constant(0.999, dtype=tf.float32)
            self.eps   = tf.constant(1e-7,  dtype=tf.float32)

            # Weight variables
            self.W, self.b = [], []
            self.mW, self.vW = [], []  # Adam moments for W
            self.mb, self.vb = [], []  # Adam moments for b
            for i in range(len(DIMS) - 1):
                fan_in = DIMS[i]
                scale  = tf.sqrt(2.0 / tf.cast(fan_in, tf.float32))
                W = tf.Variable(
                    tf.random.normal([DIMS[i], DIMS[i+1]]) * scale,
                    trainable=True, dtype=tf.float32,
                    name=f'W{i}')
                b = tf.Variable(
                    tf.zeros([DIMS[i+1]]),
                    trainable=True, dtype=tf.float32,
                    name=f'b{i}')
                self.W.append(W)
                self.b.append(b)
                self.mW.append(tf.Variable(tf.zeros_like(W), trainable=False, name=f'mW{i}'))
                self.vW.append(tf.Variable(tf.zeros_like(W), trainable=False, name=f'vW{i}'))
                self.mb.append(tf.Variable(tf.zeros_like(b), trainable=False, name=f'mb{i}'))
                self.vb.append(tf.Variable(tf.zeros_like(b), trainable=False, name=f'vb{i}'))

        def _sigmoid(self, x):
            return tf.math.sigmoid(x)

        def _sigmoid_grad(self, s):
            # s is already sigmoid(x), gradient = s*(1-s)
            return s * (1.0 - s)

        def _forward(self, x):
            """Forward pass. Returns (output, list of pre-activations, list of activations)."""
            pre = []
            act = [x]
            h = x
            for i, (W, b) in enumerate(zip(self.W, self.b)):
                z = tf.matmul(h, W) + b
                pre.append(z)
                if i < len(self.W) - 1:
                    h = self._sigmoid(z)
                else:
                    h = z   # linear output layer
                act.append(h)
            return h, pre, act

        def _adam_update(self, var, mvar, vvar, grad, lr_t):
            mvar.assign(self.b1 * mvar + (1.0 - self.b1) * grad)
            vvar.assign(self.b2 * vvar + (1.0 - self.b2) * tf.square(grad))
            var.assign_sub(lr_t * mvar / (tf.sqrt(vvar) + self.eps))

        # ── TRAIN ─────────────────────────────────────────────────────────────
        @tf.function(input_signature=[
            tf.TensorSpec([None, INPUT_DIM], tf.float32, name='x'),
        ])
        def train(self, x):
            """Manual forward + backward pass. No GradientTape."""
            batch_size = tf.cast(tf.shape(x)[0], tf.float32)

            # Forward
            output, pre, act = self._forward(x)

            # MSE loss: mean over batch and features
            diff = output - x
            loss = tf.reduce_mean(tf.square(diff))

            # Adam bias-corrected LR
            self.step.assign_add(1)
            t    = tf.cast(self.step, tf.float32)
            b1t  = tf.pow(self.b1, t)
            b2t  = tf.pow(self.b2, t)
            lr_t = self.lr * tf.sqrt(1.0 - b2t) / (1.0 - b1t)

            # Backward — manual chain rule
            # Output layer gradient (linear, no activation grad needed)
            d = (2.0 / batch_size) * diff   # (batch, OUTPUT_DIM)

            for i in reversed(range(len(self.W))):
                dW = tf.matmul(tf.transpose(act[i]), d)
                db = tf.reduce_sum(d, axis=0)
                self._adam_update(self.W[i], self.mW[i], self.vW[i], dW, lr_t)
                self._adam_update(self.b[i], self.mb[i], self.vb[i], db, lr_t)
                if i > 0:
                    # Propagate through sigmoid activation
                    d = tf.matmul(d, tf.transpose(self.W[i]))
                    d = d * self._sigmoid_grad(act[i])

            return {"loss": loss}

        # ── INFER ─────────────────────────────────────────────────────────────
        @tf.function(input_signature=[
            tf.TensorSpec([None, INPUT_DIM], tf.float32, name='x'),
        ])
        def infer(self, x):
            """Reconstruction + per-sample MSE anomaly score."""
            output, _, _ = self._forward(x)
            mse = tf.reduce_mean(tf.square(x - output), axis=1)
            return {
                "output":               output,
                "reconstruction_error": mse,
            }

    # ── Convert to TFLite ────────────────────────────────────────────────────
    module = CANAutoencoder()

    # Warm up
    dummy = tf.zeros([1, INPUT_DIM])
    module.train(dummy)
    module.infer(dummy)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [
            module.train.get_concrete_function(),
            module.infer.get_concrete_function(),
        ],
        module,
    )
    # Only TFLITE_BUILTINS — no Flex delegate needed
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
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
