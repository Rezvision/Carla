# fed_client.py  –  Federated Learning Edge Client (Raspberry Pi)
"""
Runs on each Raspberry Pi.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  PHASE 1 — Bootstrap                                │
  │  Loads weights from checkpoint or starts fresh      │
  ├─────────────────────────────────────────────────────┤
  │  PHASE 2 — Local pre-training                       │
  │  GRU autoencoder trains on local CAN data           │
  │  Anomaly detection DISABLED until threshold         │
  │  calibrated on real normal data (no false positives)│
  ├─────────────────────────────────────────────────────┤
  │  PHASE 3 — Federation                               │
  │  Sends weight arrays to server (NOT raw data)       │
  │  Receives FedAvg-averaged weights from server       │
  │  Continues local training on top of global model    │
  ├─────────────────────────────────────────────────────┤
  │  PHASE 4 — Continuous improvement                   │
  │  Rollback to best checkpoint if loss increases      │
  │  Hybrid trigger: federate on weight drift OR        │
  │  every 6 hours                                      │
  └─────────────────────────────────────────────────────┘

Model (seq2seq GRU autoencoder):
  Encoder : GRU over 20 input timesteps  (batch, 20, 8) → h_enc (batch, 32)
  Decoder : GRU unrolled for 20 steps, seeded with h_enc, autoregressive
            each step: GRU([prev_out, h]) → out_t (batch, 8)
  Loss    : MSE at every timestep — full sequence reconstruction
  Training: pure numpy BPTT through encoder + decoder (no TFLite C++)
  Infer   : pure numpy forward pass


Requirements (Raspberry Pi):
  pip install paho-mqtt numpy python-can
"""

import os
# os.environ["TFLITE_DISABLE_XNNPACK"] = "1"   # before tflite import
import json
import time
import struct
import pickle
import queue
import shutil
import threading
import numpy as np
from collections import deque

import paho.mqtt.client as mqtt

# # ── TFLite import (optional, kept for model file validation) ──────────────────
# try:
#     import tflite_runtime.interpreter as tflite
#     TFLITE_OK = True
# except ImportError:
#     try:
#         import tensorflow.lite as tflite
#         TFLITE_OK = True
#     except ImportError:
#         print("[Warning] tflite-runtime not found — model file validation disabled")
#         TFLITE_OK = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_DIM            = 160        # WINDOW_SIZE × N_FEATURES (flat)
WINDOW_SIZE          = 20
N_FEATURES           = 8
GRU_HIDDEN           = 32        # GRU hidden units (shared encoder & decoder)
LOCAL_TRAIN_INTERVAL = 300       # seconds between local training cycles
MIN_BUFFER_FRAMES    = 200       # min CAN frames before first training
LOCAL_EPOCHS         = 3         # epochs per local training cycle
BATCH_SIZE           = 32
LR                   = 0.001     # SGD learning rate
ANOMALY_PERCENTILE   = 99        # percentile used for threshold calibration
ANOMALY_SAFETY_MULT  = 1.5       # multiply p99 threshold by this — extra headroom
GRAD_CLIP_NORM       = 5.0       # global gradient norm clip (prevents score explosion)
ROLLBACK_PATIENCE    = 3         # consecutive worsening rounds before rollback
FED_BASE_INTERVAL    = 21600     # 6 hours base federation interval
FED_MIN_INTERVAL     = 1800      # 30 min minimum between federation rounds
DIVERGENCE_THRESHOLD = 0.10      # weight-drift trigger for early federation
NOISE_STD = 0.05
_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR      = os.path.join(_BASE_DIR, "models")
CHECKPOINT_DIR = "/tmp/fed_ids_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: GRU Autoencoder  (pure numpy, thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class GRUAutoencoder:
    """
    Sequence-to-sequence GRU autoencoder — symmetric encoder/decoder.

    Encoder: GRU unrolled over 20 input timesteps  → last hidden state h_enc
    Decoder: GRU unrolled for 20 output timesteps, initialised with h_enc.
             At each step the previous reconstructed frame is fed as input
             (free-running), so the decoder generates the full sequence
             conditioned purely on the compressed hidden state.
    Output:  linear projection Wo per timestep → N_FEATURES values

    Why GRU decoder is better than MLP decoder for this task:
      • Reconstruction is timestep-aware — anomaly can be localised
      • Gradient flows through the full sequence, not a single matrix multiply
      • Symmetric architecture; encoder and decoder share the same inductive bias
      • More sensitive to temporal pattern deviations (e.g. sudden CAN floods)

    
    """

    # Encoder weights (7)  +  decoder weights (9)  =  16 total
    _WEIGHT_KEYS = (
        'Wr',     'br',                          # encoder reset gate
        'Wz',     'bz',                          # encoder update gate
        'Wn_x',   'Wn_h',  'bn',                # encoder new gate
        'Dec_Wr', 'Dec_br',                      # decoder reset gate
        'Dec_Wz', 'Dec_bz',                      # decoder update gate
        'Dec_Wn_x', 'Dec_Wn_h', 'Dec_bn',       # decoder new gate
        'Wo',     'bo',                          # output projection
    )

    # Adam hyperparameters
    _BETA1 = 0.9
    _BETA2 = 0.999
    _EPS   = 1e-8

    def __init__(self):
        self.threshold  = float('inf')
        self.calibrated = False
        self.is_loaded  = True
        self._lock      = threading.Lock()
        # self._tflite_interp = None
        self._init_weights()
        self._init_adam()

    # ── Weight init ──────────────────────────────────────────────────────────

    def _init_weights(self):
        F, H = N_FEATURES, GRU_HIDDEN
        xh_enc = F + H   # encoder: [x_t, h_enc]  concatenated input
        xh_dec = F + H   # decoder: [prev_out, h_dec] — same sizes

        def w(r, c):
            return (np.random.randn(r, c) * np.sqrt(2.0 / (r + c))).astype(np.float32)
        def b(n):
            return np.zeros(n, dtype=np.float32)

        # Encoder GRU
        self.Wr    = w(xh_enc, H); self.br    = b(H)
        self.Wz    = w(xh_enc, H); self.bz    = b(H)
        self.Wn_x  = w(F,      H); self.Wn_h  = w(H, H); self.bn  = b(H)

        # Decoder GRU  (same shape as encoder — symmetric)
        self.Dec_Wr   = w(xh_dec, H); self.Dec_br   = b(H)
        self.Dec_Wz   = w(xh_dec, H); self.Dec_bz   = b(H)
        self.Dec_Wn_x = w(F,      H); self.Dec_Wn_h = w(H, H); self.Dec_bn = b(H)

        # Output projection: hidden → one reconstructed frame
        self.Wo = w(H, F); self.bo = b(F)

    # ── Adam optimiser ───────────────────────────────────────────────────────

    def _init_adam(self):
        """Zero first/second moment buffers for all weight matrices."""
        self._adam_m = {k: np.zeros_like(getattr(self, k)) for k in self._WEIGHT_KEYS}
        self._adam_v = {k: np.zeros_like(getattr(self, k)) for k in self._WEIGHT_KEYS}
        self._adam_t = 0

    def _adam_step(self, grads: dict):
        """
        Adam update for every key in grads.  Must be called inside self._lock.
        grads: {weight_key: gradient_array}
        """
        self._adam_t += 1
        t   = self._adam_t
        bc1 = 1.0 - self._BETA1 ** t
        bc2 = 1.0 - self._BETA2 ** t
        for k, g in grads.items():
            self._adam_m[k] = self._BETA1 * self._adam_m[k] + (1 - self._BETA1) * g
            self._adam_v[k] = self._BETA2 * self._adam_v[k] + (1 - self._BETA2) * (g ** 2)
            m_hat = self._adam_m[k] / bc1
            v_hat = self._adam_v[k] / bc2
            w = getattr(self, k)
            w -= LR * m_hat / (np.sqrt(v_hat) + self._EPS)

    # ── TFLite file validation (weights stay numpy) ───────────────────────────

    # def load_from_bytes(self, tflite_bytes: bytes) -> bool:
    #     path = os.path.join(MODEL_DIR, "current_model.tflite")
    #     with open(path, "wb") as f:
    #         f.write(tflite_bytes)
    #     return self._validate_tflite(path)

    # def _validate_tflite(self, path: str) -> bool:
    #     if not TFLITE_OK or not os.path.exists(path):
    #         return True
    #     try:
    #         interp = tflite.Interpreter(model_path=path, num_threads=1)
    #         interp.allocate_tensors()
    #         self._tflite_interp = interp
    #         print(f"[Model] TFLite file validated: {path}")
    #     except Exception as e:
    #         print(f"[Model] TFLite validation warning: {e} — numpy weights unaffected")
    #     return True

    # ── Encoder forward (GRU) ────────────────────────────────────────────────

    def _enc_forward(self, x_seq: np.ndarray):
        """x_seq: (batch, WINDOW_SIZE, F) → h_enc: (batch, H), cache list"""
        batch = x_seq.shape[0]
        h     = np.zeros((batch, GRU_HIDDEN), dtype=np.float32)
        cache = []
        for t in range(WINDOW_SIZE):
            x  = x_seq[:, t, :]
            xh = np.concatenate([x, h], axis=1)
            r  = _sig(xh @ self.Wr + self.br)
            z  = _sig(xh @ self.Wz + self.bz)
            n  = np.tanh(x @ self.Wn_x + (r * h) @ self.Wn_h + self.bn)
            h_new = (1.0 - z) * h + z * n
            cache.append((x, h, xh, r, z, n))
            h = h_new
        return h, cache

    # ── Decoder forward (GRU, free-running) ──────────────────────────────────

    def _dec_forward(self, h_enc: np.ndarray):
        """
        h_enc: (batch, H) — initialises decoder hidden state.
        Generates WINDOW_SIZE frames autoregressively.
        Returns recon: (batch, WINDOW_SIZE, F) and per-step cache.
        """
        batch    = h_enc.shape[0]
        h        = h_enc.copy()
        prev_out = np.zeros((batch, N_FEATURES), dtype=np.float32)
        outputs  = []
        cache    = []
        for _ in range(WINDOW_SIZE):
            xh    = np.concatenate([prev_out, h], axis=1)
            r     = _sig(xh @ self.Dec_Wr + self.Dec_br)
            z     = _sig(xh @ self.Dec_Wz + self.Dec_bz)
            n     = np.tanh(prev_out @ self.Dec_Wn_x
                            + (r * h) @ self.Dec_Wn_h + self.Dec_bn)
            h_new = (1.0 - z) * h + z * n
            out_t = h_new @ self.Wo + self.bo          # (batch, F)
            cache.append((prev_out, h, xh, r, z, n, h_new))
            outputs.append(out_t)
            prev_out = out_t
            h        = h_new
        return np.stack(outputs, axis=1), cache        # (batch, T, F)

    # ── Encoder backward (BPTT) ──────────────────────────────────────────────

    def _enc_backward(self, d_h_enc, cache):
        """Given gradient w.r.t. encoder's last hidden state, return weight grads."""
        dWr = np.zeros_like(self.Wr);   dbr = np.zeros_like(self.br)
        dWz = np.zeros_like(self.Wz);   dbz = np.zeros_like(self.bz)
        dWn_x = np.zeros_like(self.Wn_x)
        dWn_h = np.zeros_like(self.Wn_h)
        dbn   = np.zeros_like(self.bn)

        dh = d_h_enc
        for t in reversed(range(WINDOW_SIZE)):
            x, h_prev, xh, r, z, n = cache[t]
            dh_prev  = dh * (1.0 - z)
            d_z_pre  = dh * (n - h_prev) * z * (1.0 - z)
            d_n      = dh * z * (1.0 - n ** 2)

            dWn_x   += x.T @ d_n
            dbn     += d_n.sum(axis=0)
            d_rh     = d_n @ self.Wn_h.T
            dWn_h   += (r * h_prev).T @ d_n
            d_r_post = d_rh * h_prev
            dh_prev += d_rh * r

            d_r_pre  = d_r_post * r * (1.0 - r)
            dWr     += xh.T @ d_r_pre
            dbr     += d_r_pre.sum(axis=0)
            dh_prev += (d_r_pre @ self.Wr.T)[:, N_FEATURES:]

            dWz     += xh.T @ d_z_pre
            dbz     += d_z_pre.sum(axis=0)
            dh_prev += (d_z_pre @ self.Wz.T)[:, N_FEATURES:]

            dh = dh_prev
        return dWr, dbr, dWz, dbz, dWn_x, dWn_h, dbn

    # ── Decoder forward — teacher-forced (training only) ─────────────────────

    def _dec_forward_train(self, h_enc: np.ndarray, x_seq: np.ndarray):
        """
        Teacher-forced decoder for training.
        Input at step t = actual frame x_seq[:,t-1,:] (zeros at t=0).
        This gives a clean, independent gradient at every timestep — no error
        compounding across steps, which makes BPTT well-conditioned.
        Used only during train_step; inference always uses _dec_forward (free-run).
        """
        batch = h_enc.shape[0]
        h     = h_enc.copy()
        outputs, cache = [], []
        for t in range(WINDOW_SIZE):
            prev_in = (np.zeros((batch, N_FEATURES), dtype=np.float32)
                       if t == 0 else x_seq[:, t - 1, :])
            xh    = np.concatenate([prev_in, h], axis=1)
            r     = _sig(xh @ self.Dec_Wr + self.Dec_br)
            z     = _sig(xh @ self.Dec_Wz + self.Dec_bz)
            n     = np.tanh(prev_in @ self.Dec_Wn_x
                            + (r * h) @ self.Dec_Wn_h + self.Dec_bn)
            h_new = (1.0 - z) * h + z * n
            out_t = h_new @ self.Wo + self.bo
            cache.append((prev_in, h, xh, r, z, n, h_new))
            outputs.append(out_t)
            h = h_new
        return np.stack(outputs, axis=1), cache    # (batch, T, F)

    # ── Decoder backward — teacher-forced BPTT ───────────────────────────────

    def _dec_backward_tf(self, d_recon, cache):
        """
        Backward through the teacher-forced decoder.
        prev_in at each step is ground-truth data, so its gradient is discarded —
        d_prev_out does NOT accumulate across timesteps.  Only d_h flows back.
        This is what makes the gradient signal per-timestep instead of entangled.
        Returns d_h_enc (passed into encoder backward) + all decoder weight grads.
        """
        batch    = d_recon.shape[0]
        dDec_Wr  = np.zeros_like(self.Dec_Wr)
        dDec_Wz  = np.zeros_like(self.Dec_Wz)
        dDec_Wn_x = np.zeros_like(self.Dec_Wn_x)
        dDec_Wn_h = np.zeros_like(self.Dec_Wn_h)
        dDec_br  = np.zeros_like(self.Dec_br)
        dDec_bz  = np.zeros_like(self.Dec_bz)
        dDec_bn  = np.zeros_like(self.Dec_bn)
        dWo      = np.zeros_like(self.Wo)
        dbo      = np.zeros_like(self.bo)
        d_h_next = np.zeros((batch, GRU_HIDDEN), dtype=np.float32)

        for t in reversed(range(WINDOW_SIZE)):
            prev_in, h_prev, xh, r, z, n, h_new = cache[t]

            # Loss gradient at this timestep only — no chained d_prev_out
            d_out_t = d_recon[:, t, :]
            dWo    += h_new.T @ d_out_t
            dbo    += d_out_t.sum(axis=0)
            d_h_new = d_out_t @ self.Wo.T + d_h_next

            dh_prev  = d_h_new * (1.0 - z)
            d_z_pre  = d_h_new * (n - h_prev) * z * (1.0 - z)
            d_n      = d_h_new * z * (1.0 - n ** 2)

            dDec_Wn_x += prev_in.T @ d_n
            dDec_bn   += d_n.sum(axis=0)
            d_rh       = d_n @ self.Dec_Wn_h.T
            dDec_Wn_h += (r * h_prev).T @ d_n
            dh_prev   += d_rh * r
            d_r_pre    = d_rh * h_prev * r * (1.0 - r)

            dDec_Wr   += xh.T @ d_r_pre
            dDec_br   += d_r_pre.sum(axis=0)
            dh_prev   += (d_r_pre @ self.Dec_Wr.T)[:, N_FEATURES:]
            # input-dimension grad discarded — prev_in is ground truth, not network output

            dDec_Wz   += xh.T @ d_z_pre
            dDec_bz   += d_z_pre.sum(axis=0)
            dh_prev   += (d_z_pre @ self.Dec_Wz.T)[:, N_FEATURES:]

            d_h_next = dh_prev

        return (d_h_next,
                dDec_Wr, dDec_br, dDec_Wz, dDec_bz,
                dDec_Wn_x, dDec_Wn_h, dDec_bn, dWo, dbo)

    # ── Training ─────────────────────────────────────────────────────────────

    def train_step(self, x_batch: np.ndarray) -> float:
        """
        One Adam step with teacher-forced decoder.
        x_batch: (batch, 160) flat windows.
        """
        x_flat = x_batch.astype(np.float32)
        x_seq  = x_flat.reshape(-1, WINDOW_SIZE, N_FEATURES)

        # add noise to encoder input only — target stays clean
        noise  = np.random.normal(0, NOISE_STD,
                                x_seq.shape).astype(np.float32)
        x_noisy = x_seq + noise

        # Forward — teacher forcing during training
        h_enc, enc_cache = self._enc_forward(x_noisy)
        recon, dec_cache = self._dec_forward_train(h_enc, x_seq)   # (batch, T, F)

        diff = recon - x_seq
        loss = float(np.mean(diff ** 2))

        # Backward
        d_recon = (2.0 / x_flat.shape[0]) * diff
        (d_h_enc,
         dDec_Wr, dDec_br, dDec_Wz, dDec_bz,
         dDec_Wn_x, dDec_Wn_h, dDec_bn,
         dWo, dbo)                               = self._dec_backward_tf(d_recon, dec_cache)
        dWr, dbr, dWz, dbz, dWn_x, dWn_h, dbn  = self._enc_backward(d_h_enc, enc_cache)

        # Gradient clipping by global norm
        grads = {
            'Wr': dWr,      'br': dbr,
            'Wz': dWz,      'bz': dbz,
            'Wn_x': dWn_x,  'Wn_h': dWn_h,  'bn': dbn,
            'Dec_Wr': dDec_Wr,   'Dec_br': dDec_br,
            'Dec_Wz': dDec_Wz,   'Dec_bz': dDec_bz,
            'Dec_Wn_x': dDec_Wn_x, 'Dec_Wn_h': dDec_Wn_h, 'Dec_bn': dDec_bn,
            'Wo': dWo,      'bo': dbo,
        }
        global_norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values())))
        if global_norm > GRAD_CLIP_NORM:
            scale = GRAD_CLIP_NORM / global_norm
            grads = {k: v * scale for k, v in grads.items()}

        # Adam update
        with self._lock:
            self._adam_step(grads)

        return loss

    # ── Inference ────────────────────────────────────────────────────────────

    def _numpy_forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for a single flat (160,) window. Returns flat (160,)."""
        x_seq = x.reshape(1, WINDOW_SIZE, N_FEATURES).astype(np.float32)
        with self._lock:
            h_enc, _  = self._enc_forward(x_seq)
            recon, _  = self._dec_forward(h_enc)
        return recon.reshape(1, INPUT_DIM)

    def infer(self, x: np.ndarray):
        """Returns (is_anomaly: bool, score: float). Silent until calibrated."""
        if not self.calibrated:
            return False, 0.0
        x_flat = x.reshape(1, INPUT_DIM).astype(np.float32)
        output = self._numpy_forward(x)
        error  = float(np.mean((x_flat - output) ** 2))
        return error > self.threshold, error

    def reconstruction_error(self, x: np.ndarray) -> float:
        """Raw MSE — used during threshold calibration (bypasses gate)."""
        x_flat = x.reshape(1, INPUT_DIM).astype(np.float32)
        output = self._numpy_forward(x)
        return float(np.mean((x_flat - output) ** 2))

    # ── Threshold calibration ────────────────────────────────────────────────

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
                  f"min={arr.min():.6f} mean={arr.mean():.6f} max={arr.max():.6f})")

    # ── Checkpointing (atomic write) ─────────────────────────────────────────

    def save_checkpoint(self, name: str = "best") -> str:
        path = os.path.join(CHECKPOINT_DIR, f"{name}.npz")
        # np.savez appends .npz automatically when the path doesn't end in .npz.
        # Use a tmp name that already ends in .npz so the rename target is exact.
        tmp  = os.path.join(CHECKPOINT_DIR, f"{name}_tmp.npz")
        with self._lock:
            np.savez(tmp,
                     **{k: getattr(self, k) for k in self._WEIGHT_KEYS},
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
            print(f"[Model] Checkpoint corrupt — skipping restore: {e}")
            return
        with self._lock:
            for key in self._WEIGHT_KEYS:
                if key in d:
                    setattr(self, key, d[key].astype(np.float32))

            # ── Weight sanity guard ───────────────────────────────────────────
            # A collapsed or degenerate model has near-zero weight norms.
            # Compute the mean L2 norm across all weight matrices — a healthy
            # freshly-initialised model has norms in the range 0.5 – 5.0.
            # Near-zero means the checkpoint was saved after overfitting collapse
            # or after a bad FedAvg round averaged near-zero weights.
            sample_norm = float(np.mean([
                float(np.linalg.norm(getattr(self, k)))
                for k in self._WEIGHT_KEYS
            ]))
            if sample_norm < 0.01:
                print(f"[Model] Checkpoint weights degenerate "
                    f"(mean_norm={sample_norm:.6f}) — reinitialising fresh")
                self._init_weights()
                self._init_adam()
                self.threshold  = float('inf')
                self.calibrated = False
                return

            # ── Threshold sanity guard (existing) ────────────────────────────
            if 'threshold' in d:
                t = float(d['threshold'][0])
                if 'calibrated' in d and bool(d['calibrated'][0]):
                    if t > 1e-4:
                        self.threshold  = t
                        self.calibrated = True
                    else:
                        print(f"[Model] Checkpoint threshold={t:.6f} suspicious "
                            f"— resetting to uncalibrated")
                        self.threshold  = float('inf')
                        self.calibrated = False
                else:
                    self.threshold  = float('inf')
                    self.calibrated = False

        self._init_adam()
        print(f"[Model] Restored from: {path} "
            f"(mean_norm={sample_norm:.4f}, "
            f"calibrated={self.calibrated}, threshold={self.threshold:.6f})")
    # ── FedAvg interface ─────────────────────────────────────────────────────

    def get_weights(self) -> list:
        """Return list of all weight arrays for FedAvg aggregation."""
        with self._lock:
            return [getattr(self, k).copy() for k in self._WEIGHT_KEYS]


    def set_weights_from_fedavg(self, flat_list: list):

        """Apply server-averaged GRU weights with full shape validation."""
        if len(flat_list) != len(self._WEIGHT_KEYS):
            print(f"[Model] FedAvg count mismatch: "
                f"expected {len(self._WEIGHT_KEYS)}, got {len(flat_list)}")
            return

        # Validate every shape before touching any weight
        F, H, XH = N_FEATURES, GRU_HIDDEN, N_FEATURES + GRU_HIDDEN
        expected_shapes = {
            'Wr': (XH,H), 'br': (H,), 'Wz': (XH,H), 'bz': (H,),
            'Wn_x': (F,H), 'Wn_h': (H,H), 'bn': (H,),
            'Dec_Wr': (XH,H), 'Dec_br': (H,), 'Dec_Wz': (XH,H), 'Dec_bz': (H,),
            'Dec_Wn_x': (F,H), 'Dec_Wn_h': (H,H), 'Dec_bn': (H,),
            'Wo': (H,F), 'bo': (F,),
        }
        for key, arr in zip(self._WEIGHT_KEYS, flat_list):
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != expected_shapes[key]:
                print(f"[Model] FedAvg shape mismatch on '{key}': "
                    f"expected {expected_shapes[key]}, got {arr.shape} — aborted")
                return

        with self._lock:
            for key, arr in zip(self._WEIGHT_KEYS, flat_list):
                setattr(self, key, np.asarray(arr, dtype=np.float32))
        self._init_adam()   # reset moments — averaged weights shift loss landscape
        print(f"[Model] FedAvg weights applied ✓  ({len(flat_list)} arrays)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Rollback manager
# ─────────────────────────────────────────────────────────────────────────────

class RollbackManager:
    """
    Tracks loss history.  If loss worsens for ROLLBACK_PATIENCE consecutive
    local training cycles, restores the best checkpoint automatically.
    """

    def __init__(self, model: GRUAutoencoder):
        self.model        = model
        self.best_loss    = float('inf')
        self.worse_count  = 0
        self.loss_history = []

    def update(self, loss: float, round_label: str) -> bool:
        """Returns True if rollback was triggered."""
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
# SECTION 3: Federation trigger
# ─────────────────────────────────────────────────────────────────────────────

class FederationTrigger:
    """
    Decides when to send a weight update to the server.
      1. Weight divergence > threshold  → immediate
      2. Loss spike > 30%               → immediate (concept drift)
      3. 6 hours elapsed                → scheduled (skip if stable)
    """

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
        """Returns (should_federate: bool, reason: str)"""
        elapsed    = time.time() - self.last_fed_time
        divergence = self._divergence(current_weights)
        loss_change = (abs(current_loss - self.ref_loss) / max(self.ref_loss, 1e-8)
                       if self.ref_loss is not None else float('inf'))

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
# SECTION 4: CAN data collector
# ─────────────────────────────────────────────────────────────────────────────

class DataCollector:
    """
    Reads from CAN bus (or simulates data if unavailable).
    Maintains a rolling buffer and provides normalised sliding windows.
    """

    FEATURE_NAMES = [
        'speed_kmh', 'battery_level', 'throttle', 'brake',
        'steering',  'gear',          'location_x', 'location_y',
    ]

    def __init__(self, can_interface: str = 'can0', buffer_size: int = 2000):
        self.can_interface = can_interface
        self.buffer        = deque(maxlen=buffer_size)
        self.state         = {n: 0.0 for n in self.FEATURE_NAMES}
        self.running       = False
        self.frame_count      = 0
        self.last_count_reset = time.time()
        self.mean          = np.zeros(N_FEATURES, dtype=np.float32)
        self.std           = np.ones(N_FEATURES,  dtype=np.float32)
        self.scaler_fitted = False

    def start(self):
        self.running = True
        threading.Thread(target=self._can_listener, daemon=True).start()
        print(f"[DataCollector] Started on {self.can_interface}")

    def stop(self):
        self.running = False

    def _can_listener(self):
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

    def _process(self, msg):
        cid = msg.arbitration_id
        try:
            v = struct.unpack('<f', msg.data[:4])[0]
            m = {0x123: 'speed_kmh',  0x124: 'battery_level',
                 0x125: 'throttle',   0x126: 'brake',
                 0x127: 'steering',   0x128: 'gear'}
            if cid in m:
                self.state[m[cid]] = v
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
        """Return (N, 160) normalised sliding-window array for training."""
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
        """Return (160,) most recent normalised window for inference."""
        if len(self.buffer) < WINDOW_SIZE:
            return None
        if not self.scaler_fitted:
            self.fit_scaler()
        data = self._normalise(
            np.array(list(self.buffer)[-WINDOW_SIZE:], dtype=np.float32))
        return data.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Main federated client
# ─────────────────────────────────────────────────────────────────────────────

class FederatedClient:
    """Orchestrates the full federated learning lifecycle on the Pi."""

    def __init__(self, client_id: str, broker: str,
                 port: int = 1883, can_interface: str = 'can0'):
        self.client_id = client_id
        self.broker    = broker
        self.port      = port

        self.model     = GRUAutoencoder()
        self.collector = DataCollector(can_interface)
        self.rollback  = RollbackManager(self.model)
        self.trigger   = FederationTrigger()

        self.mqtt          = None
        self.connected     = False
        self.current_round = 0
        self.model_queue   = queue.Queue()
        self.running       = False
        self.current_loss  = float('inf')
        self.normal_errors = deque(maxlen=500)
        self.phase         = "local_pretraining"
        self._recent_errors  = deque(maxlen=300)   # ~30s of scores at 10Hz
        self._anomaly_tick   = 0                   # throttle ✓ prints to every 5s

        # ── Threading fix ────────────────────────────────────────────────────
        # A non-blocking lock ensures only ONE _training_cycle runs at a time.
        # Concurrent spawns (from timer + server trigger) silently drop instead
        # of piling up and corrupting shared weight arrays.
        self._train_lock      = threading.Lock()
        # Set last_train_time to now so first cycle fires after one interval,
        # not immediately on startup (avoids training before buffer fills).
        self._last_train_time = time.time()

        self._try_load_saved_model()

    def _try_load_saved_model(self):
        """Restore weights + calibration state from last run."""
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
        try:
            if message.topic == "federated/model/global":
                self.model_queue.put(pickle.loads(message.payload))
            elif message.topic == "federated/aggregation/trigger":
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

  # REPLACE the entire _apply_global_model method (lines 855–871) with:

    def _apply_global_model(self, data: dict):
        """Apply FedAvg-averaged GRU weight arrays from server."""
        round_num      = data.get("round", self.current_round + 1)
        fedavg_weights = data.get("fedavg_weights")
        server_keys    = data.get("weight_keys")   # new field from server

        if fedavg_weights is None:
            print("[Client] Global model payload missing 'fedavg_weights' — ignored")
            return

        # Verify key order matches before applying — catches any future schema drift
        if server_keys is not None:
            expected = list(self.model._WEIGHT_KEYS)
            if server_keys != expected:
                print(f"[Client] Weight key mismatch from server!\n"
                    f"  server : {server_keys}\n"
                    f"  client : {expected}\n"
                    f"  — global model NOT applied")
                return

        self.model.set_weights_from_fedavg(fedavg_weights)

        # After receiving a global model the threshold may no longer
        # reflect the new weight distribution — reset calibration so it
        # recalibrates cleanly on the next training cycle
        self.model.calibrated = False
        self.normal_errors.clear()
        print(f"[Client] Threshold reset — will recalibrate after next training cycle")

        self.current_round = round_num
        self.phase         = "federated_training"   # more accurate than local_pretraining
        self.trigger.store_reference(self.model.get_weights(), self.current_loss)
        print(f"[Client] Global model applied ✓  round={round_num}, "
            f"weights={len(fedavg_weights)}")
    # ── Training cycle ────────────────────────────────────────────────────────

    def _training_cycle(self):
        """
        Outer wrapper: acquires non-blocking lock so concurrent calls
        (from timer and server trigger) are silently dropped.
        """
        if not self._train_lock.acquire(blocking=False):
            return   # another cycle is already running — skip
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
                  f"({len(self.collector.buffer)}/{MIN_BUFFER_FRAMES} frames)")
            return

        self._publish_status("training")
        windows = self.collector.get_windows(stride=20)
        if windows is None:
            self._publish_status("idle")
            return

        # ── Local training ────────────────────────────────────────────────
        losses = []
        for _ in range(LOCAL_EPOCHS):
            idx = np.random.permutation(len(windows))
            for i in range(0, len(windows), BATCH_SIZE):
                batch = windows[idx[i:i + BATCH_SIZE]]
                if len(batch) == 0:
                    continue
                losses.append(self.model.train_step(batch))

        avg_loss = float(np.mean(losses)) if losses else float('inf')
        self.current_loss = avg_loss
        print(f"[Client {self.client_id}] "
              f"Training: loss={avg_loss:.6f}, "
              f"windows={len(windows)}, phase={self.phase}")

        # ── Rollback check ────────────────────────────────────────────────
        if self.rollback.update(avg_loss, f"round_{self.current_round}"):
            self._publish_status("rolled_back")
            return   # weights reverted — don't federate

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
        # Collect reconstruction errors on normal training data.
        # Uses raw error (bypasses calibration gate) so we can gather
        # enough data to set the threshold for the first time.
        for w in windows[:100]:
            self.normal_errors.append(self.model.reconstruction_error(w))
        self.model.calibrate_threshold(list(self.normal_errors))

        self._publish_status("idle")

    def _send_weight_update(self, weights: list, loss: float,
                             num_samples: int):
        """Send GRU weight arrays (NOT raw CAN data) to server for FedAvg."""
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

    # ── Anomaly detection ────────────────────────────────────────────────────

    def _detect_anomaly(self):
        """
        Runs every 100ms in the main loop.
        Silent (no alerts) during pretraining — calibrated=False gates MQTT publish.
        Reconstruction scores are always collected and logged (throttled for normals).
        """
        window = self.collector.get_latest_window()
        if window is None:
            return

        # Always get raw score (bypasses calibration gate)
        score = self.model.reconstruction_error(window)
        self._recent_errors.append(score)
        self._anomaly_tick += 1

        if not self.model.calibrated:
            return   # don't alert, but score was still collected above

        thr        = self.model.threshold
        is_anomaly = score > thr

        if is_anomaly:
            print(f"Anomaly Detected, score={score:.6f}  threshold={thr:.6f}  "
                  f"({score/thr*100:.0f}% of threshold)")
            if self.connected:
                self.mqtt.publish("federated/alerts/attack", json.dumps({
                    "client_id": self.client_id,
                    "score":     score,
                    "threshold": thr,
                    "round":     self.current_round,
                    "timestamp": time.time(),
                }))
        elif self._anomaly_tick % 50 == 0:
            # Log normal scores every ~5s so progress is visible without flooding
            print(f"Healthy system,  score={score:.6f}  threshold={thr:.6f}  "
                  f"({score/thr*100:.0f}% of threshold)")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print(f"FEDERATED IDS CLIENT — {self.client_id}")
        print(f"  Broker        : {self.broker}:{self.port}")
        print(f"  CAN interface : {self.collector.can_interface}")
        print(f"  Model         : GRU({GRU_HIDDEN}) autoencoder  [pure numpy]")
        # print(f"  TFLite        : {'available' if TFLITE_OK else 'not found'}"
        #       f"  [used for file validation only]")
        print(f"  Fed interval  : {FED_BASE_INTERVAL/3600:.0f}h base")
        print("=" * 60)

        self.collector.start()
        self.connect()
        time.sleep(2)
        self.running = True
        log_tick = 0

        try:
            while self.running:
                # Apply any received global model updates
                try:
                    self._apply_global_model(self.model_queue.get_nowait())
                except queue.Empty:
                    pass

                # Periodic local training (lock ensures only one runs at a time)
                if (time.time() - self._last_train_time >= LOCAL_TRAIN_INTERVAL
                        and self.collector.ready()):
                    threading.Thread(target=self._training_cycle,
                                     daemon=True).start()

                # Real-time anomaly detection (every 100ms, gated on calibration)
                self._detect_anomaly()

                # Status log every ~30s
                log_tick += 1
                if log_tick % 300 == 0:
                    _now     = time.time()
                    _elapsed = _now - self.collector.last_count_reset
                    _fps     = self.collector.frame_count / max(_elapsed, 1)
                    self.collector.frame_count      = 0
                    self.collector.last_count_reset = _now

                    # Reconstruction error stats over the last ~30s
                    if self._recent_errors:
                        _errs       = np.array(self._recent_errors, dtype=np.float32)
                        _thr        = self.model.threshold
                        _n_above    = int(np.sum(_errs > _thr))
                        _err_stats  = (f"recon: min={_errs.min():.4f} "
                                       f"mean={_errs.mean():.4f} "
                                       f"max={_errs.max():.4f} "
                                       f"above_thr={_n_above}/{len(_errs)}")
                    else:
                        _err_stats = "recon: no data yet"

                    print(f"[Client {self.client_id}] "
                          f"phase={self.phase} | "
                          f"round={self.current_round} | "
                          f"loss={self.current_loss:.6f} | "
                          f"calibrated={self.model.calibrated} | "
                          f"threshold={self.model.threshold:.4f} | "
                          f"can_fps={_fps:.1f} | "
                          f"buffer={len(self.collector.buffer)} | "
                          f"{_err_stats}")

                time.sleep(0.1)

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
    p = argparse.ArgumentParser(description="Federated IDS Edge Client")
    p.add_argument("--client-id",     default="edge_1")
    p.add_argument("--broker",        required=True,
                   help="MQTT broker IP (Windows server IP)")
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
