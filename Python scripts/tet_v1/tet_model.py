#!/usr/bin/env python3
"""
tet_model.py — Temporal Embedding Transformer autoencoder for unsupervised CAN IDS.

Core architecture from:

    L. Tao, Z. Xiyang, "Spatial-Temporal Cooperative In-Vehicle Network Intrusion
    Detection Method Based on Federated Learning", IEEE Access, 2025.

Paper FL-TET (Sec. IV):
    AE        : compress CAN features (paper: 12 → 6)
    TET       : Transformer with *temporal embedding* (TE) replacing positional encoding
                TE(p, 2i)   = sin(ts_p / 10000^(2i/d))
                TE(p, 2i+1) = cos(ts_p / 10000^(2i/d))   (Eq. 6)
    Head      : supervised multi-class FC classifier
    Training  : federated FedAvg across vehicle clients

ADAPTATION TO THIS REPOSITORY
------------------------------
Centralised unsupervised reconstruction on *translated* CAN (same contract as mvp_v1 /
vae_v1 / fsmn_ae_v1):

  1. No federated learning — train end-to-end on pooled parquet.
  2. Input is 8 decoded continuous signals (speed, battery, throttle, brake, steering,
     gear, location_x, location_y), not raw CAN ID + DATA bytes. AE compresses
     8 → 4 (same 50% ratio as the paper's 12 → 6).
  3. Window / stride = 20 / 20 (repo standard). Timestamps for TE are normalised
     row indices in [0, 1] when a real timestamp column is absent.
  4. Detection is unsupervised reconstruction MSE with dynamic threshold
     theta = mu + k*sigma (same protocol as fsmn_ae_v1), not the paper's supervised
     attack-class FC head — matching the other models' evaluation setup.
  5. Optional Gaussian input noise (paper denoising AE, sigma=0.1) during training.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax import serialization
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        'Install deps:  pip install "jax[cpu]" flax optax scikit-learn pandas pyarrow'
    ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (repo standards + paper-inspired lightweight TET)
# ──────────────────────────────────────────────────────────────────────────────

WINDOW_SIZE = 20
STRIDE = 20

FEATURES = ("speed_kmh", "battery_level", "throttle", "brake",
            "steering", "gear", "location_x", "location_y")
N_FEATURES = len(FEATURES)

# AE: 50% compression (paper 12→6; here 8→4).
AE_LATENT = 4
AE_HIDDEN = 16

# TET Transformer.
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 128
DROPOUT_RATE = 0.1

# Denoising AE noise (paper Sec. IV-C, sigma=0.1).
NOISE_STD = 0.1

# Dynamic anomaly threshold theta = mu + k*sigma.
THRESHOLD_K = 2.5
GRAD_CLIP_NORM = 5.0


@dataclasses.dataclass
class TETConfig:
    window: int = WINDOW_SIZE
    n_features: int = N_FEATURES
    ae_latent: int = AE_LATENT
    ae_hidden: int = AE_HIDDEN
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    ff_dim: int = FF_DIM
    dropout: float = DROPOUT_RATE
    noise_std: float = NOISE_STD

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TETConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ──────────────────────────────────────────────────────────────────────────────
# Temporal embedding (paper Eq. 6)
# ──────────────────────────────────────────────────────────────────────────────

def temporal_embedding(ts: jnp.ndarray, d_model: int) -> jnp.ndarray:
    """
    ts: (B, T) normalised timestamps.
    returns TE: (B, T, d_model)  with sin/cos of ts / 10000^(2i/d)
    """
    half = d_model // 2
    i = jnp.arange(half, dtype=jnp.float32)
    freqs = 10000.0 ** (2.0 * i / d_model)
    angles = ts[..., None] / freqs                      # (B, T, half)
    te = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if d_model % 2 == 1:                                # pad if odd width
        te = jnp.pad(te, ((0, 0), (0, 0), (0, 1)))
    return te


# ──────────────────────────────────────────────────────────────────────────────
# Flax modules
# ──────────────────────────────────────────────────────────────────────────────

class FeatureAE(nn.Module):
    """Per-timestep autoencoder: F → hidden → latent → hidden → F."""
    n_features: int
    ae_latent: int
    ae_hidden: int

    def setup(self):
        self.enc1 = nn.Dense(self.ae_hidden, name="ae_enc1")
        self.enc2 = nn.Dense(self.ae_latent, name="ae_enc2")
        self.dec1 = nn.Dense(self.ae_hidden, name="ae_dec1")
        self.dec2 = nn.Dense(self.n_features, name="ae_dec2")

    def encode(self, x):                                 # (B, T, F) → (B, T, L)
        h = nn.relu(self.enc1(x))
        return nn.relu(self.enc2(h))

    def decode(self, z):                                 # (B, T, L) → (B, T, F)
        h = nn.relu(self.dec1(z))
        return self.dec2(h)                              # linear out (z-scored input)

    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z), z


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    ff_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Pre-norm multi-head self-attention + residual.
        h = nn.LayerNorm(name="ln1")(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout,
            name="mha",
        )(h, h, deterministic=deterministic)
        x = x + nn.Dropout(self.dropout, name="drop1")(h, deterministic=deterministic)

        # Pre-norm feed-forward + residual.
        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(self.ff_dim, name="ff1")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model, name="ff2")(h)
        h = nn.Dropout(self.dropout, name="drop2")(h, deterministic=deterministic)
        return x + h


class TETNet(nn.Module):
    """
    AE feature compression + Temporal Embedding Transformer + AE decode.

    Forward:
        x (B,T,F), ts (B,T)
          → AE.encode → z (B,T,L)
          → Dense(d_model) + TE(ts)
          → N Transformer blocks
          → Dense(L) → AE.decode → recon (B,T,F)
    """
    n_features: int
    ae_latent: int
    ae_hidden: int
    d_model: int
    n_heads: int
    n_layers: int
    ff_dim: int
    dropout: float

    def setup(self):
        self.ae = FeatureAE(self.n_features, self.ae_latent, self.ae_hidden)
        self.in_proj = nn.Dense(self.d_model, name="in_proj")
        self.blocks = [
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.dropout,
                             name=f"block_{i}")
            for i in range(self.n_layers)
        ]
        self.out_proj = nn.Dense(self.ae_latent, name="out_proj")
        self.out_norm = nn.LayerNorm(name="out_norm")

    def __call__(self, x, ts, deterministic: bool = True):
        z = self.ae.encode(x)                            # (B, T, L)
        h = self.in_proj(z) + temporal_embedding(ts, self.d_model)
        for block in self.blocks:
            h = block(h, deterministic=deterministic)
        h = self.out_norm(h)
        z_hat = self.out_proj(h)                         # (B, T, L)
        recon = self.ae.decode(z_hat)
        return recon, z

    def reconstruct(self, x, ts):
        recon, _ = self(x, ts, deterministic=True)
        return recon


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def _loss_fn(params, net, x, ts, rng, noise_std):
    rng_noise, rng_drop = jax.random.split(rng)
    if noise_std > 0.0:
        x_in = x + jax.random.normal(rng_noise, x.shape) * noise_std
    else:
        x_in = x
    recon, _ = net.apply(params, x_in, ts, deterministic=False,
                         rngs={"dropout": rng_drop})
    return jnp.mean((recon - x) ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# High-level wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TETAE:
    """Trainable AE+TET reconstruction detector (centralised)."""

    def __init__(self, config: Optional[TETConfig] = None, params=None, seed: int = 42):
        self.cfg = config or TETConfig()
        self.net = TETNet(
            n_features=self.cfg.n_features,
            ae_latent=self.cfg.ae_latent,
            ae_hidden=self.cfg.ae_hidden,
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            ff_dim=self.cfg.ff_dim,
            dropout=self.cfg.dropout,
        )
        key = jax.random.PRNGKey(seed)
        self._key = key
        if params is None:
            dummy_x = jnp.zeros((1, self.cfg.window, self.cfg.n_features))
            dummy_ts = jnp.linspace(0.0, 1.0, self.cfg.window)[None, :]
            init_key, drop_key, self._key = jax.random.split(key, 3)
            params = self.net.init(
                {"params": init_key, "dropout": drop_key},
                dummy_x, dummy_ts, deterministic=True,
            )
        self.params = params

        self.recon_threshold: float = float("inf")
        self.err_mean: float = 0.0
        self.err_std: float = 0.0
        self.k: float = THRESHOLD_K

        self._reconstruct = jax.jit(
            lambda p, x, ts: self.net.apply(p, x, ts, method=self.net.reconstruct))

    def num_params(self) -> int:
        return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(self.params)))

    def approx_size_kb(self) -> float:
        return self.num_params() * 4 / 1024.0

    def train(self, windows: np.ndarray, timestamps: np.ndarray, *,
              epochs: int = 100, batch_size: int = 1024, lr: float = 1e-3,
              weight_decay: float = 0.0, noise_std: Optional[float] = None,
              seed: int = 42, verbose: bool = True):
        """Train on standardised windows (N, T, F) and ts (N, T)."""
        windows = np.asarray(windows, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float32)
        noise_std = self.cfg.noise_std if noise_std is None else float(noise_std)
        n = len(windows)
        steps_per_epoch = max(1, n // batch_size)
        total_steps = max(1, epochs * steps_per_epoch)

        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps,
                                               alpha=0.0)
        optimizer = optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP_NORM),
            optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
        )
        opt_state = optimizer.init(self.params)
        net = self.net

        @jax.jit
        def train_step(params, opt_state, x, ts, rng):
            loss, grads = jax.value_and_grad(_loss_fn)(
                params, net, x, ts, rng, noise_std)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        rng = np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed)
        history = []
        for ep in range(epochs):
            perm = rng.permutation(n)
            losses = []
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) == 0:
                    continue
                key, sub = jax.random.split(key)
                self.params, opt_state, loss = train_step(
                    self.params, opt_state,
                    jnp.asarray(windows[idx]), jnp.asarray(timestamps[idx]), sub)
                losses.append(float(loss))
            avg = float(np.mean(losses)) if losses else float("nan")
            history.append(avg)
            if verbose:
                print(f"  epoch {ep + 1:3d}/{epochs}  loss={avg:.6f}")
        return history

    def reconstruction_error(self, windows: np.ndarray, timestamps: np.ndarray,
                             batch_size: int = 4096) -> np.ndarray:
        """Per-window MSE between input and reconstruction."""
        windows = np.asarray(windows, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float32)
        out = []
        for i in range(0, len(windows), batch_size):
            x = jnp.asarray(windows[i:i + batch_size])
            ts = jnp.asarray(timestamps[i:i + batch_size])
            recon = self._reconstruct(self.params, x, ts)
            err = np.asarray(jnp.mean((recon - x) ** 2, axis=(1, 2)))
            out.append(err)
        return np.concatenate(out, axis=0) if out else np.zeros((0,))

    def predict(self, windows: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        return (self.reconstruction_error(windows, timestamps)
                > self.recon_threshold).astype(int)

    def build_detector(self, train_windows: np.ndarray, train_ts: np.ndarray,
                       *, k: float = THRESHOLD_K):
        errs = self.reconstruction_error(train_windows, train_ts)
        self.err_mean = float(errs.mean())
        self.err_std = float(errs.std())
        self.k = float(k)
        self.recon_threshold = self.err_mean + self.k * self.err_std
        return {
            "recon_threshold": self.recon_threshold,
            "err_mean": self.err_mean,
            "err_std": self.err_std,
            "k": self.k,
        }

    def save(self, directory: str, name: str = "tet_ae"):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{name}_params.msgpack"), "wb") as f:
            f.write(serialization.to_bytes(self.params))
        meta = {
            "config": self.cfg.to_dict(),
            "recon_threshold": self.recon_threshold,
            "err_mean": self.err_mean,
            "err_std": self.err_std,
            "k": self.k,
            "num_params": self.num_params(),
        }
        with open(os.path.join(directory, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return directory

    @classmethod
    def load(cls, directory: str, name: str = "tet_ae", seed: int = 42) -> "TETAE":
        with open(os.path.join(directory, f"{name}_meta.json")) as f:
            meta = json.load(f)
        cfg = TETConfig.from_dict(meta["config"])
        model = cls(cfg, seed=seed)
        with open(os.path.join(directory, f"{name}_params.msgpack"), "rb") as f:
            model.params = serialization.from_bytes(model.params, f.read())
        model.recon_threshold = meta["recon_threshold"]
        model.err_mean = meta.get("err_mean", 0.0)
        model.err_std = meta.get("err_std", 0.0)
        model.k = meta.get("k", THRESHOLD_K)
        return model
