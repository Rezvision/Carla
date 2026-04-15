# fed_client.py  –  Federated Learning Edge Client (Raspberry Pi)
# JAX rewrite — same GRU autoencoder architecture, automatic differentiation
# replaces all manual BPTT backward passes.
#
# Key changes from numpy version:
#   • _enc_backward and _dec_backward_tf deleted — jax.grad handles this
#   • weights live in a single `params` dict passed functionally
#   • jax.jit compiles forward + train_step for ARM speedup (~3-5x after warmup)
#   • jax.numpy (jnp) replaces numpy for all array ops inside the model
#   • Adam state is a plain dict of arrays — no class needed
#   • Everything else (MQTT, rollback, federation, data collector) unchanged
#
# Install on Pi:
#   pip install "jax[cpu]" paho-mqtt python-can
#   (jax[cpu] pulls jaxlib automatically — no GPU needed)

import os
import json
import time
import struct
import pickle
import queue
import shutil
import threading
import numpy as np
from collections import deque

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from functools import partial

import paho.mqtt.client as mqtt

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_DIM            = 160
WINDOW_SIZE          = 20
N_FEATURES           = 8
GRU_HIDDEN           = 32
LOCAL_TRAIN_INTERVAL = 300
MIN_BUFFER_FRAMES    = 200
LOCAL_EPOCHS         = 3
BATCH_SIZE           = 32
LR                   = 0.001
ANOMALY_PERCENTILE   = 99
ANOMALY_SAFETY_MULT  = 1.5
GRAD_CLIP_NORM       = 5.0
NOISE_STD            = 0.05        # denoising autoencoder — prevents memorisation
L2_LAMBDA            = 1e-4        # weight regularisation
MIN_DELTA            = 0.0005      # early stopping threshold per epoch
ROLLBACK_PATIENCE    = 3
FED_BASE_INTERVAL    = 3600   # 60 min — matches server late phase
FED_MIN_INTERVAL     = 900    # 15 min — matches server early phase
DIVERGENCE_THRESHOLD = 0.10
_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = "/tmp/fed_ids_checkpoints"
MODEL_DIR      = os.path.join(_BASE_DIR, "models")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Force JAX to use CPU (Pi has no GPU) and use float32 throughout
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)   # stay in float32
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Pure functions — forward pass (jit-compiled)
# These are module-level pure functions that take params as an argument.
# JAX requires pure functions (no side effects, no mutable state) for jit/grad.
# ─────────────────────────────────────────────────────────────────────────────

def _sig(x):
    """Numerically stable sigmoid using jnp."""
    return jax.nn.sigmoid(x)          # jax.nn.sigmoid is stable by default


def _gru_step_enc(params, x, h):
    """
    Single encoder GRU step.
    x: (batch, F)   h: (batch, H)
    Returns h_new: (batch, H)
    """
    xh = jnp.concatenate([x, h], axis=1)                    # (batch, 40)
    r  = _sig(xh @ params['Wr']  + params['br'])             # reset gate
    z  = _sig(xh @ params['Wz']  + params['bz'])             # update gate
    n  = jnp.tanh(x @ params['Wn_x']
                  + (r * h) @ params['Wn_h']
                  + params['bn'])                             # candidate
    return (1.0 - z) * h + z * n                             # (batch, H)


def _gru_step_dec(params, prev_out, h):
    """
    Single decoder GRU step.
    prev_out: (batch, F)   h: (batch, H)
    Returns h_new: (batch, H),  out: (batch, F)
    """
    xh    = jnp.concatenate([prev_out, h], axis=1)
    r     = _sig(xh @ params['Dec_Wr'] + params['Dec_br'])
    z     = _sig(xh @ params['Dec_Wz'] + params['Dec_bz'])
    n     = jnp.tanh(prev_out @ params['Dec_Wn_x']
                     + (r * h) @ params['Dec_Wn_h']
                     + params['Dec_bn'])
    h_new = (1.0 - z) * h + z * n
    out   = h_new @ params['Wo'] + params['bo']               # (batch, F)
    return h_new, out


def encoder_forward(params, x_seq):
    """
    x_seq: (batch, WINDOW_SIZE, F)
    Returns h_enc: (batch, H)
    JAX scan replaces the Python for loop — compiled and fast.
    """
    batch = x_seq.shape[0]
    h     = jnp.zeros((batch, GRU_HIDDEN), dtype=jnp.float32)

    # jax.lax.scan is the JAX way to write a loop over a sequence.
    # It compiles the loop body once and runs it T times — no Python overhead.
    # carry = h (hidden state),  xs = frames at each timestep
    def step(h, x_t):
        h_new = _gru_step_enc(params, x_t, h)
        return h_new, None    # carry forward h, emit nothing

    # x_seq transposed to (WINDOW_SIZE, batch, F) so scan iterates over time
    h_enc, _ = jax.lax.scan(step, h, x_seq.transpose(1, 0, 2))
    return h_enc                                               # (batch, H)


def decoder_forward_train(params, h_enc, x_seq):
    """
    Teacher-forced decoder — used during training only.
    prev_in at step t = ground-truth frame t-1 (zeros at t=0).
    x_seq: (batch, WINDOW_SIZE, F)
    Returns recon: (batch, WINDOW_SIZE, F)
    """
    batch = h_enc.shape[0]
    h     = h_enc

    def step(h, x_prev):
        # x_prev is the ground-truth previous frame (teacher forcing)
        h_new, out = _gru_step_dec(params, x_prev, h)
        return h_new, out

    # Build teacher-forced inputs: [zeros, x[:,0,:], x[:,1,:], ..., x[:,18,:]]
    zeros      = jnp.zeros((batch, N_FEATURES), dtype=jnp.float32)
    tf_inputs  = jnp.concatenate(
        [zeros[None], x_seq.transpose(1, 0, 2)[:-1]], axis=0
    )                                                          # (T, batch, F)

    _, recon_T = jax.lax.scan(step, h, tf_inputs)             # (T, batch, F)
    return recon_T.transpose(1, 0, 2)                         # (batch, T, F)


def decoder_forward_infer(params, h_enc):
    """
    Free-running decoder — used at inference only.
    Each step feeds its own output as the next input.
    Returns recon: (batch, WINDOW_SIZE, F)
    """
    batch = h_enc.shape[0]
    h     = h_enc

    def step(carry, _):
        h, prev_out = carry
        h_new, out  = _gru_step_dec(params, prev_out, h)
        return (h_new, out), out

    init_out = jnp.zeros((batch, N_FEATURES), dtype=jnp.float32)
    _, recon_T = jax.lax.scan(step, (h, init_out), None, length=WINDOW_SIZE)
    return recon_T.transpose(1, 0, 2)                         # (batch, T, F)


def loss_fn(params, x_seq_noisy, x_seq_clean):
    """
    Full autoencoder loss: MSE(reconstruction, clean) + L2 penalty.
    x_seq_noisy: (batch, T, F) — encoder input  (with noise for denoising AE)
    x_seq_clean: (batch, T, F) — reconstruction target (original, no noise)

    This is the function JAX differentiates. Because it is pure (no side
    effects, params passed explicitly), jax.value_and_grad works on it directly.
    """
    h_enc = encoder_forward(params, x_seq_noisy)
    recon = decoder_forward_train(params, h_enc, x_seq_clean)

    # Reconstruction loss
    mse = jnp.mean((recon - x_seq_clean) ** 2)

    # L2 regularisation — sum of squared weights across all arrays
    l2 = sum(jnp.sum(v ** 2) for v in params.values())

    return mse + L2_LAMBDA * l2


# JIT-compile the loss and its gradient together.
# value_and_grad returns (loss_scalar, grad_dict) in one pass — more efficient
# than calling value and grad separately.
# argnums=0 means differentiate w.r.t. the first argument (params).
loss_and_grad = jit(value_and_grad(loss_fn, argnums=0))


@jit
def infer_reconstruction(params, x_flat):
    """
    Single-window inference. Returns MSE reconstruction error (scalar).
    x_flat: (160,) flat window
    JIT-compiled — fast after first call.
    """
    x_seq = x_flat.reshape(1, WINDOW_SIZE, N_FEATURES)
    h_enc = encoder_forward(params, x_seq)
    recon = decoder_forward_infer(params, h_enc)
    return jnp.mean((recon - x_seq) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Parameter initialisation and Adam state
# ─────────────────────────────────────────────────────────────────────────────

# Weight key list — same order as numpy version for FedAvg compatibility
_WEIGHT_KEYS = (
    'Wr',     'br',
    'Wz',     'bz',
    'Wn_x',   'Wn_h',  'bn',
    'Dec_Wr', 'Dec_br',
    'Dec_Wz', 'Dec_bz',
    'Dec_Wn_x', 'Dec_Wn_h', 'Dec_bn',
    'Wo',     'bo',
)


def init_params(key=None):
    """
    Initialise all 16 weight arrays using Glorot scaling.
    Returns a plain dict of JAX arrays.
    JAX uses explicit PRNG keys rather than global random state.
    """
    if key is None:
        key = jax.random.PRNGKey(int(time.time()) % (2**31))

    F, H, XH = N_FEATURES, GRU_HIDDEN, N_FEATURES + GRU_HIDDEN

    def w(key, r, c):
        scale = jnp.sqrt(2.0 / (r + c))
        return jax.random.normal(key, (r, c)) * scale

    def b(n):
        return jnp.zeros(n, dtype=jnp.float32)

    keys = jax.random.split(key, 12)  # 12 weight matrices need random init

    return {
        'Wr':      w(keys[0],  XH, H),   'br':      b(H),
        'Wz':      w(keys[1],  XH, H),   'bz':      b(H),
        'Wn_x':    w(keys[2],  F,  H),
        'Wn_h':    w(keys[3],  H,  H),   'bn':      b(H),
        'Dec_Wr':  w(keys[4],  XH, H),   'Dec_br':  b(H),
        'Dec_Wz':  w(keys[5],  XH, H),   'Dec_bz':  b(H),
        'Dec_Wn_x':w(keys[6],  F,  H),
        'Dec_Wn_h':w(keys[7],  H,  H),   'Dec_bn':  b(H),
        'Wo':      w(keys[8],  H,  F),   'bo':      b(F),
    }


def init_adam(params):
    """
    Initialise Adam moment buffers matching the params dict structure.
    Returns (m, v, step) where m and v are dicts of zero arrays.
    """
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}
    return m, v, 0


def adam_step(params, grads, m, v, step,
              lr=LR, b1=0.9, b2=0.999, eps=1e-8):
    """
    One Adam update. Pure function — returns updated params, m, v, step.
    No mutation — JAX style.
    """
    step  = step + 1
    bc1   = 1.0 - b1 ** step
    bc2   = 1.0 - b2 ** step

    new_m, new_v, new_params = {}, {}, {}
    for k in params:
        new_m[k] = b1 * m[k] + (1 - b1) * grads[k]
        new_v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
        m_hat    = new_m[k] / bc1
        v_hat    = new_v[k] / bc2
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    return new_params, new_m, new_v, step


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: GRUAutoencoder class — same public interface as numpy version
# Internal implementation uses JAX but all public methods (get_weights,
# set_weights_from_fedavg, save_checkpoint, restore_checkpoint, train_step,
# reconstruction_error, calibrate_threshold) have identical signatures.
# ─────────────────────────────────────────────────────────────────────────────

class GRUAutoencoder:
    """
    GRU sequence-to-sequence autoencoder backed by JAX.

    Public interface is identical to the numpy version — drop-in replacement.
    Internally:
      • params dict replaces 16 individual weight attributes
      • jax.value_and_grad replaces _enc_backward + _dec_backward_tf
      • jax.lax.scan replaces Python for loops over timesteps
      • jit compilation gives ~3-5x speedup on ARM after first call
    """

    _WEIGHT_KEYS = _WEIGHT_KEYS   # same order as numpy version

    _BETA1 = 0.9
    _BETA2 = 0.999
    _EPS   = 1e-8

    def __init__(self):
        self.threshold  = float('inf')
        self.calibrated = False
        self.is_loaded  = True
        self._lock      = threading.Lock()

        self.params       = init_params()
        self.adam_m, self.adam_v, self.adam_t = init_adam(self.params)

        # Warm up JIT — first call compiles, subsequent calls are fast.
        # Do this at init so the first real training step is not slow.
        print("[Model] Warming up JIT compilation...")
        dummy = jnp.zeros((1, WINDOW_SIZE, N_FEATURES))
        _ = loss_and_grad(self.params, dummy, dummy)
        dummy_flat = jnp.zeros((INPUT_DIM,))
        _ = infer_reconstruction(self.params, dummy_flat)
        print("[Model] JIT warmup complete")

    # ── Weight access (FedAvg interface) ─────────────────────────────────────

    def get_weights(self) -> list:
        """Return list of weight arrays in _WEIGHT_KEYS order for FedAvg."""
        with self._lock:
            return [np.array(self.params[k]) for k in self._WEIGHT_KEYS]

    def set_weights_from_fedavg(self, flat_list: list):
        """Apply server-averaged weights with shape validation."""
        if len(flat_list) != len(self._WEIGHT_KEYS):
            print(f"[Model] FedAvg count mismatch: "
                  f"expected {len(self._WEIGHT_KEYS)}, got {len(flat_list)}")
            return

        F, H, XH = N_FEATURES, GRU_HIDDEN, N_FEATURES + GRU_HIDDEN
        expected_shapes = {
            'Wr': (XH,H), 'br': (H,), 'Wz': (XH,H), 'bz': (H,),
            'Wn_x': (F,H), 'Wn_h': (H,H), 'bn': (H,),
            'Dec_Wr': (XH,H), 'Dec_br': (H,), 'Dec_Wz': (XH,H), 'Dec_bz': (H,),
            'Dec_Wn_x': (F,H), 'Dec_Wn_h': (H,H), 'Dec_bn': (H,),
            'Wo': (H,F), 'bo': (F,),
        }
        for key, arr in zip(self._WEIGHT_KEYS, flat_list):
            if np.asarray(arr).shape != expected_shapes[key]:
                print(f"[Model] FedAvg shape mismatch on '{key}' — aborted")
                return

        with self._lock:
            self.params = {
                k: jnp.array(np.asarray(arr, dtype=np.float32))
                for k, arr in zip(self._WEIGHT_KEYS, flat_list)
            }
            # Reset Adam moments — averaged weights shift the loss landscape
            self.adam_m, self.adam_v, self.adam_t = init_adam(self.params)

        print(f"[Model] FedAvg weights applied ✓  ({len(flat_list)} arrays)")

    # ── Training ─────────────────────────────────────────────────────────────

    def train_step(self, x_batch: np.ndarray) -> float:
        """
        One Adam step. x_batch: (batch, 160) flat windows.

        Steps:
          1. Add noise  → denoising autoencoder (prevents memorisation)
          2. value_and_grad  → loss scalar + full gradient dict in one pass
          3. Gradient clipping by global norm
          4. Adam update  → new params
        """
        x_flat  = jnp.array(x_batch.astype(np.float32))
        x_seq   = x_flat.reshape(-1, WINDOW_SIZE, N_FEATURES)

        # Denoising: add noise to encoder input, reconstruct clean target
        noise_key = jax.random.PRNGKey(int(time.time_ns()) % (2**31))
        noise     = jax.random.normal(noise_key, x_seq.shape) * NOISE_STD
        x_noisy   = x_seq + noise

        with self._lock:
            # value_and_grad returns (loss, grad_dict) in one compiled call
            loss_val, grads = loss_and_grad(self.params, x_noisy, x_seq)

            # Gradient clipping by global norm
            global_norm = jnp.sqrt(sum(
                jnp.sum(g ** 2) for g in grads.values()
            ))
            scale  = jnp.where(global_norm > GRAD_CLIP_NORM,
                                GRAD_CLIP_NORM / global_norm,
                                1.0)
            grads  = {k: g * scale for k, g in grads.items()}

            # Adam update — returns new params and updated moment state
            self.params, self.adam_m, self.adam_v, self.adam_t = adam_step(
                self.params, grads,
                self.adam_m, self.adam_v, self.adam_t
            )

        return float(loss_val)

    # ── Inference ─────────────────────────────────────────────────────────────

    def reconstruction_error(self, x: np.ndarray) -> float:
        """Raw MSE for a single (160,) window. JIT-compiled."""
        x_flat = jnp.array(x.reshape(INPUT_DIM).astype(np.float32))
        with self._lock:
            err = infer_reconstruction(self.params, x_flat)
        return float(err)

    def infer(self, x: np.ndarray):
        """Returns (is_anomaly: bool, score: float). Silent until calibrated."""
        if not self.calibrated:
            return False, 0.0
        score = self.reconstruction_error(x)
        return score > self.threshold, score

    # ── Threshold calibration ─────────────────────────────────────────────────

    def calibrate_threshold(self, normal_errors: list):
        if len(normal_errors) > 20:
            arr = np.array(normal_errors, dtype=np.float32)
            p99 = float(np.percentile(arr, ANOMALY_PERCENTILE))
            t   = p99 * ANOMALY_SAFETY_MULT
            with self._lock:
                self.threshold  = t
                self.calibrated = True
            print(f"[Model] Threshold calibrated: {t:.6f} "
                  f"(p{ANOMALY_PERCENTILE}={p99:.6f} × {ANOMALY_SAFETY_MULT} | "
                  f"n={len(normal_errors)} | "
                  f"min={arr.min():.6f} mean={arr.mean():.6f} "
                  f"max={arr.max():.6f})")

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, name: str = "best") -> str:
        path = os.path.join(CHECKPOINT_DIR, f"{name}.npz")
        tmp  = os.path.join(CHECKPOINT_DIR, f"{name}_tmp.npz")
        with self._lock:
            np.savez(tmp,
                     **{k: np.array(self.params[k])
                        for k in self._WEIGHT_KEYS},
                     threshold=np.array([self.threshold]),
                     calibrated=np.array([int(self.calibrated)]))
        shutil.move(tmp, path)
        print(f"[Model] Checkpoint saved: {path}")
        return path

    def restore_checkpoint(self, name: str = "best"):
        path = os.path.join(CHECKPOINT_DIR, f"{name}.npz")
        if not os.path.exists(path):
            print(f"[Model] No checkpoint at {path}")
            return
        try:
            d = np.load(path)
        except Exception as e:
            print(f"[Model] Checkpoint corrupt — skipping: {e}")
            return

        with self._lock:
            # Load weights into params dict as JAX arrays
            new_params = {}
            for key in self._WEIGHT_KEYS:
                if key in d:
                    new_params[key] = jnp.array(
                        d[key].astype(np.float32))

            # Weight sanity guard — catches collapsed/degenerate checkpoints
            sample_norm = float(np.mean([
                float(np.linalg.norm(np.array(new_params[k])))
                for k in self._WEIGHT_KEYS
            ]))
            if sample_norm < 0.01:
                print(f"[Model] Checkpoint weights degenerate "
                      f"(mean_norm={sample_norm:.6f}) — reinitialising fresh")
                self.params       = init_params()
                self.adam_m, self.adam_v, self.adam_t = init_adam(self.params)
                self.threshold    = float('inf')
                self.calibrated   = False
                return

            self.params = new_params

            # Threshold sanity guard
            if 'threshold' in d:
                t = float(d['threshold'][0])
                if 'calibrated' in d and bool(d['calibrated'][0]):
                    if t > 1e-4:
                        self.threshold  = t
                        self.calibrated = True
                    else:
                        print(f"[Model] Threshold={t:.6f} suspicious "
                              f"— resetting to uncalibrated")
                        self.threshold  = float('inf')
                        self.calibrated = False
                else:
                    self.threshold  = float('inf')
                    self.calibrated = False

            # Reset Adam — stale moments are wrong after weight change
            self.adam_m, self.adam_v, self.adam_t = init_adam(self.params)

        print(f"[Model] Restored: {path}  "
              f"(mean_norm={sample_norm:.4f}, "
              f"calibrated={self.calibrated}, "
              f"threshold={self.threshold:.6f})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RollbackManager — unchanged from numpy version
# ─────────────────────────────────────────────────────────────────────────────

class RollbackManager:
    def __init__(self, model: GRUAutoencoder):
        self.model        = model
        self.best_loss    = float('inf')
        self.worse_count  = 0
        self.loss_history = []

    def update(self, loss: float, round_label: str) -> bool:
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
        if len(self.loss_history) < 3:
            return True
        recent = self.loss_history[-3:]
        return recent[-1] <= recent[0]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FederationTrigger — unchanged from numpy version
# ─────────────────────────────────────────────────────────────────────────────

class FederationTrigger:
    def __init__(self):
        self.ref_weights       = None
        self.ref_loss          = None
        self.last_fed_time     = time.time()
        self.adaptive_interval = FED_BASE_INTERVAL

    def store_reference(self, weights: list, loss: float):
        self.ref_weights   = [w.copy() for w in weights]
        self.ref_loss      = loss
        self.last_fed_time = time.time()

    def _divergence(self, current_weights: list) -> float:
        if not self.ref_weights:
            return float('inf')
        total_delta = sum(float(np.sum((c - r) ** 2))
                          for c, r in zip(current_weights, self.ref_weights))
        total_n     = sum(c.size for c in current_weights)
        return float(np.sqrt(total_delta / max(total_n, 1)))

    def should_federate(self, current_weights: list, current_loss: float):
        elapsed     = time.time() - self.last_fed_time
        divergence  = self._divergence(current_weights)
        loss_change = (abs(current_loss - self.ref_loss) / max(self.ref_loss, 1e-8)
                       if self.ref_loss is not None else float('inf'))

        # First-ever federation: no reference weights yet, send immediately
        if self.ref_weights is None:
            return True, "initial_federation"

        if divergence > DIVERGENCE_THRESHOLD and elapsed > FED_MIN_INTERVAL:
            return True, f"weight_divergence={divergence:.4f}"
        if loss_change > 0.30 and elapsed > FED_MIN_INTERVAL:
            self.adaptive_interval = FED_BASE_INTERVAL
            return True, f"concept_drift={loss_change:.2%}"
        if elapsed >= self.adaptive_interval:
            if divergence > 0.01:
                return True, "scheduled"
            else:
                self.adaptive_interval = min(
                    self.adaptive_interval * 1.5, FED_BASE_INTERVAL * 3)
                self.last_fed_time = time.time()
                return False, "skipped_stable"
        return False, (f"waiting {elapsed/3600:.1f}h"
                       f"/{self.adaptive_interval/3600:.1f}h")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: DataCollector — unchanged from numpy version
# ─────────────────────────────────────────────────────────────────────────────

class DataCollector:
    FEATURE_NAMES = [
        'speed_kmh', 'battery_level', 'throttle', 'brake',
        'steering',  'gear',          'location_x', 'location_y',
    ]

    def __init__(self, can_interface: str = 'can0', buffer_size: int = 2000,
                 edge_num: int = 1):
        self.can_interface    = can_interface
        self.buffer           = deque(maxlen=buffer_size)
        self.state            = {n: 0.0 for n in self.FEATURE_NAMES}
        self.running          = False
        self.frame_count      = 0
        self.last_count_reset = time.time()
        self.mean             = np.zeros(N_FEATURES, dtype=np.float32)
        self.std              = np.ones(N_FEATURES,  dtype=np.float32)
        self.scaler_fitted    = False

        # ── Per-vehicle CAN ID mapping ────────────────────────────────────────
        # edge_1 → 0x123–0x128,  edge_2 → 0x223–0x228,  etc.
        # The hundreds digit encodes the vehicle/edge number.
        base_offset = edge_num * 0x100
        self.can_id_map = {
            base_offset + 0x23: 'speed_kmh',
            base_offset + 0x24: 'battery_level',
            base_offset + 0x25: 'throttle',
            base_offset + 0x26: 'brake',
            base_offset + 0x27: 'steering',
            base_offset + 0x28: 'gear',
        }
        print(f"[DataCollector] CAN ID map (edge {edge_num}): "
              f"{', '.join(f'0x{k:03X}→{v}' for k, v in self.can_id_map.items())}")

    def start(self):
        self.running = True
        threading.Thread(target=self._can_listener, daemon=True).start()
        print(f"[DataCollector] Started on {self.can_interface}")

    def stop(self):
        self.running = False

    def _can_listener(self):
        bus = None
        try:
            import can
            bus = can.Bus(channel=self.can_interface, interface='socketcan')
            while self.running:
                msg = bus.recv(timeout=1.0)
                if msg:
                    self._process(msg)
        except Exception as e:
            print(f"[DataCollector] CAN error: {e} — using simulated data")
            self._simulate()
        finally:
            if bus is not None:
                try:
                    bus.shutdown()
                except Exception:
                    pass

    def _process(self, msg):
        cid = msg.arbitration_id
        try:
            v = struct.unpack('<f', msg.data[:4])[0]
            if cid in self.can_id_map:
                self.state[self.can_id_map[cid]] = v
            self._push()
        except Exception:
            pass

    def _simulate(self):
        import random
        while self.running:
            s = self.state
            s['speed_kmh']     = max(0, s.get('speed_kmh',     50) + random.gauss(0, 2))
            s['battery_level'] = max(0, min(100, s.get('battery_level', 80) - 0.01))
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
        self.frame_count += 1

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

    def get_windows(self, stride: int = 20,
                    max_windows: int = 400) -> np.ndarray:
        """Return (N, 160) normalised sliding-window array. Stride=20 default."""
        if not self.ready():
            return None
        if not self.scaler_fitted:
            self.fit_scaler()
        data = self._normalise(np.array(list(self.buffer), dtype=np.float32))
        out  = []
        for i in range(0, min(len(data) - WINDOW_SIZE,
                              max_windows * stride), stride):
            out.append(data[i:i + WINDOW_SIZE].flatten())
        return np.array(out, dtype=np.float32) if out else None

    def get_latest_window(self) -> np.ndarray:
        if len(self.buffer) < WINDOW_SIZE:
            return None
        if not self.scaler_fitted:
            self.fit_scaler()
        data = self._normalise(
            np.array(list(self.buffer)[-WINDOW_SIZE:], dtype=np.float32))
        return data.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: FederatedClient — same as numpy version with early stopping added
# ─────────────────────────────────────────────────────────────────────────────

class FederatedClient:
    def __init__(self, client_id: str, broker: str,
                 port: int = 1883, can_interface: str = 'can0'):
        self.client_id = client_id
        self.broker    = broker
        self.port      = port

        self.model     = GRUAutoencoder()
        try:
            edge_num = int(client_id.split('_')[-1])
        except (ValueError, IndexError):
            edge_num = 1
        self.collector = DataCollector(can_interface, edge_num=edge_num)
        self.rollback  = RollbackManager(self.model)
        self.trigger   = FederationTrigger()

        self.mqtt               = None
        self.connected          = False
        self.current_round      = 0
        self.model_queue        = queue.Queue()
        self.running            = False
        self.current_loss       = float('inf')
        self.normal_errors      = deque(maxlen=500)
        self.phase              = "local_pretraining"
        self._recent_errors     = deque(maxlen=300)
        self._anomaly_tick      = 0
        self._server_requested  = False
        self._train_lock        = threading.Lock()
        self._last_train_time   = time.time()
        self._last_applied_round = -1   # dedup retained MQTT messages

        self._try_load_saved_model()

    def _try_load_saved_model(self):
        for name in ("shutdown", "best"):
            path = os.path.join(CHECKPOINT_DIR, f"{name}.npz")
            if os.path.exists(path):
                self.model.restore_checkpoint(name)
                print(f"[Client] Restored from '{name}' checkpoint")
                return

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
        self.mqtt.connect(self.broker, self.port, keepalive=120)
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
        try:
            if message.topic == "federated/model/global":
                self.model_queue.put(pickle.loads(message.payload))
            elif message.topic == "federated/aggregation/trigger":
                print(f"[Client {self.client_id}] Server requested training round")
                self._server_requested = True
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
        round_num = data.get("round", self.current_round + 1)

        # Dedup — retained MQTT messages are re-delivered on every reconnect
        if round_num <= self._last_applied_round:
            print(f"[Client] Global model round={round_num} "
                  f"already applied — skipping")
            return

        fedavg_weights = data.get("fedavg_weights")
        server_keys    = data.get("weight_keys")

        if fedavg_weights is None:
            print("[Client] Payload missing 'fedavg_weights' — ignored")
            return

        if server_keys is not None:
            if server_keys != list(self.model._WEIGHT_KEYS):
                print(f"[Client] Weight key mismatch — global model NOT applied")
                return

        self.model.set_weights_from_fedavg(fedavg_weights)
        self.model.calibrated = False
        self.normal_errors.clear()

        self.current_round        = round_num
        self._last_applied_round  = round_num
        self.phase                = "federated_training"
        self.trigger.store_reference(self.model.get_weights(),
                                     self.current_loss)
        print(f"[Client] Global model applied ✓  round={round_num}")

    # ── Training cycle ─────────────────────────────────────────────────────────

    def _training_cycle(self):
        if not self._train_lock.acquire(blocking=False):
            return
        try:
            self._run_training()
        except Exception as e:
            print(f"[Client] Training error: {e}")
        finally:
            self._train_lock.release()
            self._last_train_time = time.time()

    def _run_training(self):
        if not self.collector.ready():
            print(f"[Client] Buffer not ready "
                  f"({len(self.collector.buffer)}/{MIN_BUFFER_FRAMES})")
            return

        self._publish_status("training")
        windows = self.collector.get_windows(stride=20)   # non-overlapping
        if windows is None:
            self._publish_status("idle")
            return

        # ── Local training with early stopping ───────────────────────────
        losses     = []
        prev_loss  = float('inf')

        for epoch in range(LOCAL_EPOCHS):
            epoch_losses = []
            idx = np.random.permutation(len(windows))
            for i in range(0, len(windows), BATCH_SIZE):
                batch = windows[idx[i:i + BATCH_SIZE]]
                if len(batch) == 0:
                    continue
                epoch_losses.append(self.model.train_step(batch))

            epoch_avg   = float(np.mean(epoch_losses))
            improvement = prev_loss - epoch_avg

            if epoch > 0 and improvement < MIN_DELTA:
                print(f"[Client] Early stop at epoch {epoch + 1}: "
                      f"improvement={improvement:.6f} < {MIN_DELTA}")
                losses.extend(epoch_losses)
                break

            prev_loss = epoch_avg
            losses.extend(epoch_losses)

        avg_loss          = float(np.mean(losses)) if losses else float('inf')
        self.current_loss = avg_loss
        print(f"[Client {self.client_id}] "
              f"Training: loss={avg_loss:.6f}, "
              f"windows={len(windows)}, phase={self.phase}")

        # ── Rollback check ────────────────────────────────────────────────
        if self.rollback.update(avg_loss, f"round_{self.current_round}"):
            self._publish_status("rolled_back")
            return

        # ── Federation trigger ────────────────────────────────────────────
        current_weights = self.model.get_weights()
        server_req = self._server_requested
        self._server_requested = False   # consume before any early return

        if server_req:
            print(f"[Client {self.client_id}] Server-requested send — bypassing trigger")
            self._send_weight_update(current_weights, avg_loss, len(windows))
            self.trigger.store_reference(current_weights, avg_loss)
        else:
            should_fed, reason = self.trigger.should_federate(
                current_weights, avg_loss)
            if should_fed and self.rollback.is_improving():
                self._send_weight_update(current_weights, avg_loss, len(windows))
                self.trigger.store_reference(current_weights, avg_loss)
            else:
                print(f"[Client] Federation: {reason}")

        # ── Threshold calibration ─────────────────────────────────────────
        for w in windows[:100]:
            self.normal_errors.append(self.model.reconstruction_error(w))
        self.model.calibrate_threshold(list(self.normal_errors))

        self._publish_status("idle")

    def _send_weight_update(self, weights: list, loss: float,
                             num_samples: int):
        if not self.connected:
            print("[Client] Not connected — weight update skipped")
            return
        payload = pickle.dumps({
            "client_id":   self.client_id,
            "round":       self.current_round,
            "weights":     weights,
            "num_samples": num_samples,
            "loss":        loss,
            "timestamp":   time.time(),
        })
        self.mqtt.publish("federated/model/updates", payload, qos=1)
        print(f"[Client {self.client_id}] Sent weight update: "
              f"round={self.current_round}, loss={loss:.6f}, "
              f"samples={num_samples}, size={len(payload)/1024:.1f} KB")

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def _detect_anomaly(self):
        window = self.collector.get_latest_window()
        if window is None:
            return

        score = self.model.reconstruction_error(window)
        self._recent_errors.append(score)
        self._anomaly_tick += 1

        if not self.model.calibrated:
            return

        thr        = self.model.threshold
        is_anomaly = score > thr

        if is_anomaly:
            print(f"Anomaly Detected  score={score:.6f}  "
                  f"threshold={thr:.6f}  ({score/thr*100:.0f}%)")
            if self.connected:
                self.mqtt.publish("federated/alerts/attack", json.dumps({
                    "client_id": self.client_id,
                    "score":     score,
                    "threshold": thr,
                    "round":     self.current_round,
                    "timestamp": time.time(),
                }))
        elif self._anomaly_tick % 50 == 0:
            print(f"Healthy system    score={score:.6f}  "
                  f"threshold={thr:.6f}  ({score/thr*100:.0f}%)")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print(f"FEDERATED IDS CLIENT — {self.client_id}")
        print(f"  Broker        : {self.broker}:{self.port}")
        print(f"  CAN interface : {self.collector.can_interface}")
        print(f"  Model         : GRU({GRU_HIDDEN}) autoencoder  [JAX/JIT]")
        print(f"  Fed interval  : {FED_BASE_INTERVAL/3600:.0f}h base")
        print(f"  Stride        : 20 (non-overlapping windows)")
        print(f"  Noise std     : {NOISE_STD} (denoising autoencoder)")
        print("=" * 60)

        self.collector.start()
        self.connect()
        time.sleep(2)
        self.running = True
        log_tick     = 0

        try:
            while self.running:
                try:
                    self._apply_global_model(self.model_queue.get_nowait())
                except queue.Empty:
                    pass

                if (time.time() - self._last_train_time >= LOCAL_TRAIN_INTERVAL
                        and self.collector.ready()):
                    threading.Thread(target=self._training_cycle,
                                     daemon=True).start()

                self._detect_anomaly()

                log_tick += 1
                if log_tick % 300 == 0:
                    _now     = time.time()
                    _elapsed = _now - self.collector.last_count_reset
                    _fps     = self.collector.frame_count / max(_elapsed, 1)
                    self.collector.frame_count      = 0
                    self.collector.last_count_reset = _now

                    if self._recent_errors:
                        _errs    = np.array(self._recent_errors,
                                            dtype=np.float32)
                        _thr     = self.model.threshold
                        _n_above = int(np.sum(_errs > _thr))
                        _stats   = (f"recon: min={_errs.min():.4f} "
                                    f"mean={_errs.mean():.4f} "
                                    f"max={_errs.max():.4f} "
                                    f"above_thr={_n_above}/{len(_errs)}")
                    else:
                        _stats = "recon: no data yet"

                    print(f"[Client {self.client_id}] "
                          f"phase={self.phase} | "
                          f"round={self.current_round} | "
                          f"loss={self.current_loss:.6f} | "
                          f"calibrated={self.model.calibrated} | "
                          f"threshold={self.model.threshold:.4f} | "
                          f"can_fps={_fps:.1f} | "
                          f"buffer={len(self.collector.buffer)} | "
                          f"{_stats}")

                time.sleep(0.05)   # 50ms — gives MQTT thread more CPU time

        except KeyboardInterrupt:
            print(f"\n[Client {self.client_id}] Shutting down...")
        finally:
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
    p = argparse.ArgumentParser(description="Federated IDS Edge Client (JAX)")
    p.add_argument("--client-id",     default="edge_1")
    p.add_argument("--broker",        required=True)
    p.add_argument("--port",          type=int, default=1883)
    p.add_argument("--can-interface", default="can0")
    args = p.parse_args()

    FederatedClient(
        client_id=args.client_id,
        broker=args.broker,
        port=args.port,
        can_interface=args.can_interface,
    ).run()


if __name__ == "__main__":
    main()
