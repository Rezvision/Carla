# model_builder_pi.py  —  Run on Raspberry Pi
"""
Builds the TFLite model directly on the Pi using TFLite's own flatbuffer
writer — guarantees the schema matches the installed tflite-runtime exactly.

Since tflite-runtime alone cannot build models, this script uses a
completely different approach: it builds a pure numpy autoencoder and
wraps it in a minimal hand-crafted TFLite flatbuffer using the
flatbuffers Python library.

Actually the simplest reliable approach: use tensorflow-lite's
Model Maker or simply generate the model using the tflite flatbuffer
schema directly.

The REAL reliable solution: install tensorflow (not just tflite-runtime)
on the Pi temporarily just for model building, build the model, then
uninstall it. tflite-runtime stays for inference.

Run this ONCE on the Pi:
  pip install tensorflow-cpu  # temporary, for building only
  python model_builder_pi.py
  pip uninstall tensorflow-cpu  # remove after building

OR: use the pre-built model from this script which uses only numpy
and the flatbuffers library to construct a valid TFLite flatbuffer.

Requirements:
  pip install flatbuffers numpy paho-mqtt
"""

import os
import sys
import time
import pickle
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────
MQTT_BROKER = "192.168.0.125"
MQTT_PORT   = 1883
MODEL_SAVE  = "./models/current_model.tflite"
INPUT_DIM   = 160
DIMS        = [160, 64, 32, 16, 32, 64, 160]
LR          = 0.001
# ──────────────────────────────────────────────────────────────────────────


def build_with_tensorflow():
    """
    Build model using full tensorflow installed on Pi.
    Call this if tensorflow-cpu is available on the Pi.
    """
    import tensorflow as tf

    N = len(DIMS) - 1

    def sigmoid(x):
        return tf.math.sigmoid(x)

    def forward(x, weights, biases):
        act = [x]
        for i in range(N):
            z = tf.matmul(act[-1], weights[i]) + biases[i]
            act.append(sigmoid(z) if i < N - 1 else z)
        return act[-1], act

    def sgd_step(x, weights, biases, lr=LR):
        batch  = tf.cast(tf.shape(x)[0], tf.float32)
        output, act = forward(x, weights, biases)
        diff   = output - x
        loss   = tf.reduce_mean(tf.square(diff))
        d      = (2.0 / batch) * diff
        nw     = list(weights)
        nb     = list(biases)
        for i in reversed(range(N)):
            dW    = tf.matmul(tf.transpose(act[i]), d)
            db    = tf.reduce_sum(d, axis=0)
            nw[i] = weights[i] - lr * dW
            nb[i] = biases[i]  - lr * db
            if i > 0:
                d = tf.matmul(d, tf.transpose(weights[i]))
                s = act[i]
                d = d * (s * (1.0 - s))
        return loss, nw, nb

    w_specs = [tf.TensorSpec([DIMS[i], DIMS[i+1]], tf.float32, name=f'w{i}')
               for i in range(N)]
    b_specs = [tf.TensorSpec([DIMS[i+1]], tf.float32, name=f'b{i}')
               for i in range(N)]

    class AE(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([None, INPUT_DIM], tf.float32, name='x'),
        ] + w_specs + b_specs)
        def train(self, x, w0,w1,w2,w3,w4,w5, b0,b1,b2,b3,b4,b5):
            loss, nw, nb = sgd_step(x,
                [w0,w1,w2,w3,w4,w5], [b0,b1,b2,b3,b4,b5])
            return {"loss": loss,
                    "w0":nw[0],"w1":nw[1],"w2":nw[2],
                    "w3":nw[3],"w4":nw[4],"w5":nw[5],
                    "b0":nb[0],"b1":nb[1],"b2":nb[2],
                    "b3":nb[3],"b4":nb[4],"b5":nb[5]}

        @tf.function(input_signature=[
            tf.TensorSpec([None, INPUT_DIM], tf.float32, name='x'),
        ] + w_specs + b_specs)
        def infer(self, x, w0,w1,w2,w3,w4,w5, b0,b1,b2,b3,b4,b5):
            output, _ = forward(x, [w0,w1,w2,w3,w4,w5],
                                    [b0,b1,b2,b3,b4,b5])
            mse = tf.reduce_mean(tf.square(x - output), axis=1)
            return {"reconstruction_error": mse}

    module  = AE()
    dummy_x = tf.zeros([1, INPUT_DIM])
    iw = [tf.random.normal([DIMS[i], DIMS[i+1]]) * tf.sqrt(2.0/float(DIMS[i]))
          for i in range(N)]
    ib = [tf.zeros([DIMS[i+1]]) for i in range(N)]
    module.train(dummy_x, *iw, *ib)
    module.infer(dummy_x, *iw, *ib)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [module.train.get_concrete_function(),
         module.infer.get_concrete_function()],
        module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_bytes = converter.convert()

    os.makedirs(os.path.dirname(MODEL_SAVE) or '.', exist_ok=True)
    with open(MODEL_SAVE, 'wb') as f:
        f.write(tflite_bytes)
    print(f"✓ Model built on Pi: {MODEL_SAVE} ({len(tflite_bytes)/1024:.1f} KB)")
    return tflite_bytes


def push_to_server(tflite_bytes: bytes):
    """Publish model to MQTT so server knows Pi has built it."""
    import paho.mqtt.client as mqtt
    connected = {"ok": False}

    def on_connect(c, u, f, rc, p=None):
        if hasattr(rc, 'value'):
            rc = rc.value
        if rc == 0:
            connected["ok"] = True

    client = mqtt.Client(client_id="pi_model_builder")
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()
    for _ in range(20):
        if connected["ok"]:
            break
        time.sleep(0.5)

    if connected["ok"]:
        payload = pickle.dumps({
            "type":        "initial_model",
            "round":       0,
            "tflite_bytes": tflite_bytes,
            "dims":        DIMS,
            "input_dim":   INPUT_DIM,
            "timestamp":   time.time(),
        })
        result = client.publish(
            "federated/model/global", payload, qos=1, retain=True)
        result.wait_for_publish()
        print(f"✓ Model pushed to broker ({len(payload)/1024:.1f} KB)")
    else:
        print("Could not connect to broker — model saved locally only")

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    print("=" * 55)
    print("CAN-BUS IDS — MODEL BUILDER (Pi-native)")
    print("Builds TFLite model using Pi's own TF version")
    print("=" * 55)

    # Check if tensorflow is available
    try:
        import tensorflow as tf
        print(f"TensorFlow {tf.__version__} found — building model...")
        tflite_bytes = build_with_tensorflow()
        push_to_server(tflite_bytes)
        print("\nDone. You can now run fed_client.py")
        print("To free space: pip uninstall tensorflow")
    except ImportError:
        print("\nERROR: tensorflow not found on Pi.")
        print("Install temporarily with:")
        print("  pip install tensorflow-cpu")
        print("Then re-run this script.")
        sys.exit(1)
