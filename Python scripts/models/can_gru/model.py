#!/usr/bin/env python3
"""
can_gru — GRU autoencoder for *raw* CAN frames.

Combines:
  1. Chowdhury VAE feature engineering (Sec. 3.1.1): binary arbitration ID →
     continuous bits, payload bytes, temporal channels  (see ``features.py``)
  2. A learnable ID embedding projection over those bits
  3. The existing seq2seq GRU-AE reconstruction backbone (``models.gru``)

This is a *separate* model from the decoded-telemetry ``gru`` used by CARLA /
Kaggle.  Input windows are shaped ``(N, T, N_RAW_FEATURES)`` with the first
``ID_BITS`` columns holding the ID bit field.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
except ImportError as e:  # pragma: no cover
    raise SystemExit('Install deps: pip install "jax[cpu]"') from e

from .features import (
    EMBED_DIM,
    FEATURES,
    ID_BITS,
    N_MODEL_FEATURES,
    N_RAW_FEATURES,
)

# Paper-aligned defaults for raw CAN (VAE used window 50).
WINDOW_SIZE = 50
STRIDE = 50
GRU_HIDDEN = 64
ANOMALY_PERCENTILE = 99
ANOMALY_SAFETY_MULT = 1.0
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
GRAD_CLIP_NORM = 5.0
L2_LAMBDA = 1e-4
LR = 1e-3
NOISE_STD = 0.05

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)

_WEIGHT_KEYS = (
    "W_embed", "b_embed",
    "Wr", "br", "Wz", "bz", "Wn_x", "Wn_h", "bn",
    "Dec_Wr", "Dec_br", "Dec_Wz", "Dec_bz", "Dec_Wn_x", "Dec_Wn_h", "Dec_bn",
    "Wo", "bo",
)


def _sig(x):
    return jax.nn.sigmoid(x)


def _embed(params, x_raw):
    """(B, T, N_RAW) → (B, T, N_MODEL) via ID-bit embedding + passthrough."""
    id_bits = x_raw[..., :ID_BITS]
    rest = x_raw[..., ID_BITS:]
    emb = id_bits @ params["W_embed"] + params["b_embed"]
    return jnp.concatenate([emb, rest], axis=-1)


def _gru_step_enc(params, x, h):
    xh = jnp.concatenate([x, h], axis=1)
    r = _sig(xh @ params["Wr"] + params["br"])
    z = _sig(xh @ params["Wz"] + params["bz"])
    n = jnp.tanh(x @ params["Wn_x"] + (r * h) @ params["Wn_h"] + params["bn"])
    return (1.0 - z) * h + z * n


def _gru_step_dec(params, prev_out, h):
    xh = jnp.concatenate([prev_out, h], axis=1)
    r = _sig(xh @ params["Dec_Wr"] + params["Dec_br"])
    z = _sig(xh @ params["Dec_Wz"] + params["Dec_bz"])
    n = jnp.tanh(
        prev_out @ params["Dec_Wn_x"] + (r * h) @ params["Dec_Wn_h"] + params["Dec_bn"]
    )
    h_new = (1.0 - z) * h + z * n
    out = h_new @ params["Wo"] + params["bo"]
    return h_new, out


def _encoder_forward(params, x_seq, hidden: int):
    batch = x_seq.shape[0]
    h = jnp.zeros((batch, hidden), dtype=jnp.float32)

    def step(h, x_t):
        return _gru_step_enc(params, x_t, h), None

    h_enc, _ = jax.lax.scan(step, h, x_seq.transpose(1, 0, 2))
    return h_enc


def _decoder_forward_train(params, h_enc, x_seq, n_features: int):
    batch = h_enc.shape[0]
    h = h_enc

    def step(h, x_prev):
        return _gru_step_dec(params, x_prev, h)

    zeros = jnp.zeros((batch, n_features), dtype=jnp.float32)
    tf_inputs = jnp.concatenate(
        [zeros[None], x_seq.transpose(1, 0, 2)[:-1]], axis=0
    )
    _, recon_T = jax.lax.scan(step, h, tf_inputs)
    return recon_T.transpose(1, 0, 2)


def _decoder_forward_infer(params, h_enc, n_features: int, window: int):
    batch = h_enc.shape[0]
    h = h_enc

    def step(carry, _):
        h, prev_out = carry
        h_new, out = _gru_step_dec(params, prev_out, h)
        return (h_new, out), out

    init_out = jnp.zeros((batch, n_features), dtype=jnp.float32)
    _, recon_T = jax.lax.scan(step, (h, init_out), None, length=window)
    return recon_T.transpose(1, 0, 2)


def _loss_fn(params, x_noisy_raw, x_clean_raw, hidden: int, n_model: int):
    x_noisy = _embed(params, x_noisy_raw)
    x_clean = _embed(params, x_clean_raw)
    h_enc = _encoder_forward(params, x_noisy, hidden)
    recon = _decoder_forward_train(params, h_enc, x_clean, n_model)
    mse = jnp.mean((recon - x_clean) ** 2)
    l2 = sum(jnp.sum(v ** 2) for v in params.values())
    return mse + L2_LAMBDA * l2


def _init_params(key, n_raw: int, n_model: int, hidden: int, embed_dim: int):
    rest_dim = n_raw - ID_BITS
    assert n_model == embed_dim + rest_dim

    F, H, XH = n_model, hidden, n_model + hidden

    def w(k, r, c):
        scale = jnp.sqrt(2.0 / (r + c))
        return jax.random.normal(k, (r, c)) * scale

    def b(n):
        return jnp.zeros(n, dtype=jnp.float32)

    keys = jax.random.split(key, 14)
    return {
        "W_embed": w(keys[0], ID_BITS, embed_dim),
        "b_embed": b(embed_dim),
        "Wr": w(keys[1], XH, H), "br": b(H),
        "Wz": w(keys[2], XH, H), "bz": b(H),
        "Wn_x": w(keys[3], F, H),
        "Wn_h": w(keys[4], H, H), "bn": b(H),
        "Dec_Wr": w(keys[5], XH, H), "Dec_br": b(H),
        "Dec_Wz": w(keys[6], XH, H), "Dec_bz": b(H),
        "Dec_Wn_x": w(keys[7], F, H),
        "Dec_Wn_h": w(keys[8], H, H), "Dec_bn": b(H),
        "Wo": w(keys[9], H, F), "bo": b(F),
    }


def _init_adam(params):
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}
    return m, v, 0


@jax.jit
def _adam_step(params, grads, m, v, step, lr=LR):
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = step + 1
    bc1 = 1.0 - b1 ** step
    bc2 = 1.0 - b2 ** step
    new_m, new_v, new_params = {}, {}, {}
    for k in params:
        new_m[k] = b1 * m[k] + (1 - b1) * grads[k]
        new_v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
        m_hat = new_m[k] / bc1
        v_hat = new_v[k] / bc2
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return new_params, new_m, new_v, step


class CANGRUAutoencoder:
    """Raw-CAN GRU-AE with Chowdhury ID embedding + temporal features."""

    features = FEATURES

    def __init__(
        self,
        n_features: int = N_RAW_FEATURES,
        window_size: int = WINDOW_SIZE,
        hidden: int = GRU_HIDDEN,
        embed_dim: int = EMBED_DIM,
        seed: int = 42,
        checkpoint_dir: Optional[Path] = None,
    ):
        if n_features != N_RAW_FEATURES:
            raise ValueError(
                f"can_gru expects n_features={N_RAW_FEATURES} (engineered raw CAN), "
                f"got {n_features}"
            )
        self.n_features = int(n_features)          # raw / profile dim
        self.n_model_features = N_MODEL_FEATURES   # after ID embedding
        self.window_size = int(window_size)
        self.hidden = int(hidden)
        self.embed_dim = int(embed_dim)
        self.input_dim = self.window_size * self.n_features
        self.checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.threshold = float("inf")
        self.calibrated = False

        key = jax.random.PRNGKey(seed)
        self.params = _init_params(
            key, self.n_features, self.n_model_features, self.hidden, self.embed_dim
        )
        self.adam_m, self.adam_v, self.adam_t = _init_adam(self.params)
        self.rng_key = jax.random.PRNGKey(seed + 1)

        self._loss_and_grad = jit(
            value_and_grad(
                lambda p, xn, xc: _loss_fn(
                    p, xn, xc, self.hidden, self.n_model_features
                )
            )
        )

        def _infer_fn(p, x_flat):
            x_raw = x_flat.reshape(1, self.window_size, self.n_features)
            x = _embed(p, x_raw)
            h = _encoder_forward(p, x, self.hidden)
            recon = _decoder_forward_infer(
                p, h, self.n_model_features, self.window_size
            )
            return jnp.mean((recon - x) ** 2)

        self._infer = jit(_infer_fn)

        dummy = jnp.zeros((1, self.window_size, self.n_features))
        _ = self._loss_and_grad(self.params, dummy, dummy)
        _ = self._infer(self.params, jnp.zeros((self.input_dim,)))

    def train_step(self, x_batch: np.ndarray) -> float:
        x_flat = jnp.array(x_batch.astype(np.float32))
        if x_flat.ndim == 2 and x_flat.shape[-1] == self.input_dim:
            x_seq = x_flat.reshape(-1, self.window_size, self.n_features)
        elif x_flat.ndim == 3:
            x_seq = x_flat
        else:
            raise ValueError(f"Bad batch shape {tuple(x_flat.shape)}")
        self.rng_key, sub = jax.random.split(self.rng_key)
        noise = jax.random.normal(sub, x_seq.shape) * NOISE_STD
        # Do not noise the discrete ID bit field.
        noise = noise.at[..., :ID_BITS].set(0.0)
        x_noisy = x_seq + noise

        loss_val, grads = self._loss_and_grad(self.params, x_noisy, x_seq)
        global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads.values()))
        scale = jnp.where(
            global_norm > GRAD_CLIP_NORM, GRAD_CLIP_NORM / global_norm, 1.0
        )
        grads = {k: g * scale for k, g in grads.items()}
        self.params, self.adam_m, self.adam_v, self.adam_t = _adam_step(
            self.params, grads, self.adam_m, self.adam_v, self.adam_t, LR,
        )
        return float(loss_val)

    def reconstruction_error(self, x: np.ndarray) -> float:
        x_flat = jnp.array(x.reshape(self.input_dim).astype(np.float32))
        return float(self._infer(self.params, x_flat))

    def reconstruction_errors(self, windows_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [self.reconstruction_error(w) for w in windows_flat], dtype=np.float32
        )

    def calibrate_threshold(
        self,
        normal_errors: list | np.ndarray,
        *,
        percentile: float = ANOMALY_PERCENTILE,
        safety_mult: float = ANOMALY_SAFETY_MULT,
    ):
        arr = np.asarray(normal_errors, dtype=np.float32)
        if len(arr) <= 20:
            return
        p = float(np.percentile(arr, percentile))
        self.threshold = p * safety_mult
        self.calibrated = True
        print(
            f"[Model] threshold={self.threshold:.6f} "
            f"(p{percentile}={p:.6f} × {safety_mult})"
        )

    def save_checkpoint(self, name: str = "can_gru") -> Path:
        path = self.checkpoint_dir / f"{name}.npz"
        tmp = self.checkpoint_dir / f"{name}_tmp.npz"
        np.savez(
            tmp,
            **{k: np.array(self.params[k]) for k in _WEIGHT_KEYS},
            threshold=np.array([self.threshold]),
            calibrated=np.array([int(self.calibrated)]),
            n_features=np.array([self.n_features]),
            window_size=np.array([self.window_size]),
            hidden=np.array([self.hidden]),
            embed_dim=np.array([self.embed_dim]),
        )
        shutil.move(str(tmp), str(path))
        print(f"[Model] checkpoint: {path}")
        return path

    def restore_checkpoint(self, name: str = "can_gru"):
        path = self.checkpoint_dir / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        d = np.load(path)
        self.params = {k: jnp.array(d[k]) for k in _WEIGHT_KEYS}
        self.threshold = float(d["threshold"][0])
        self.calibrated = bool(int(d["calibrated"][0]))
        self.adam_m, self.adam_v, self.adam_t = _init_adam(self.params)
        print(f"[Model] restored: {path}")
