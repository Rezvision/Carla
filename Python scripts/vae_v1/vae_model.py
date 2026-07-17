#!/usr/bin/env python3
"""
vae_model.py — LSTM Variational Autoencoder for unsupervised CAN intrusion detection.

Implementation of the model proposed in:
    S. Chowdhury, "Edge-Deployable Unsupervised Intrusion Detection for CAN Using
    Variational Autoencoders and Latent Space Disentanglement", KTH / Scania, 2025.

The paper's architecture (Sec. 3.2, Fig. 3.3) is reproduced here in JAX/Flax:

    Encoder :  LSTM(ReLU)  ->  global average pooling  ->  Dense(mu), Dense(logvar)
               logvar clipped to [-10, 5], std = softplus(logvar)
    Sampling:  reparameterisation trick with multi-sample Monte-Carlo (n=5 train, n=1 infer)
    Decoder :  RepeatVector  ->  LSTM(ReLU)  ->  TimeDistributed Dense (no output activation)
    Loss    :  beta-VAE  ->  MSE reconstruction  +  beta * KL divergence

Three anomaly-detection heads are provided (Sec. 3.3):
    1. reconstruction error  (needs encoder + decoder)
    2. latent distance       (encoder only, BallTree nearest-neighbour)  <- paper Alg. 1/2
    3. latent clustering      (encoder only, Isolation Forest)            <- paper Alg. 3/4

ADAPTATION TO THIS DATASET
--------------------------
The paper's input is raw CAN (arbitration ID + 8 payload bytes) plus engineered
temporal features, with binary arbitration IDs mapped to a continuous space via an
embedding layer (Sec. 3.1.1). This repository's data is *already translated* CAN —
decoded continuous signals (speed, battery, throttle, brake, steering, gear,
location_x, location_y). Those signals are continuous, so the binary->continuous
embedding step is unnecessary and is omitted; the decoded signals are fed directly
into a message window aligned with mvp_v1 (WINDOW_SIZE=20, STRIDE=20) and
z-score standardised before the model. (The VAE paper used window 50; we use 20
so all models in this repo are comparable on the same inputs.)
"""
from __future__ import annotations

import dataclasses
import json
import os
from functools import partial
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

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import BallTree
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install deps:  pip install scikit-learn") from e


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (paper Table 3.1)
# ──────────────────────────────────────────────────────────────────────────────

# Message-based window: 20 consecutive messages, stride 20 (matches mvp_v1).
WINDOW_SIZE = 20
STRIDE = 20

# Decoded-CAN signal channels used as model input.
FEATURES = ("speed_kmh", "battery_level", "throttle", "brake",
            "steering", "gear", "location_x", "location_y")
N_FEATURES = len(FEATURES)

# β-VAE settings.
LATENT_DIM = 10                 # Table 3.1
LSTM_UNITS = 64                 # encoder/decoder hidden width (unspecified in paper)
BETA_RECON = 0.8                # β for reconstruction-tuned model
BETA_LATENT = 2.0               # β for latent-space-tuned model
N_SAMPLES_TRAIN = 5             # multi-sampling Monte-Carlo estimator (Sec. 3.2.2)
N_SAMPLES_INFER = 1

# Numerical-stability clip on the encoder variance head (Sec. 3.2).
LOGVAR_MIN = -10.0
LOGVAR_MAX = 5.0

# Isolation Forest (Table 3.1).
IF_ESTIMATORS = 100
IF_CONTAMINATION = 5e-3
IF_RANDOM_STATE = 42

# Threshold margin factor γ (paper Alg. 1, line 13: τ = max(dist) · (1 + γ)).
THRESHOLD_MARGIN = 0.0


# Recurrent activation.
#
# The paper (Sec. 3.2) states the LSTM uses ReLU. In practice a ReLU LSTM has an
# unbounded cell state and its forward activations can explode over multi-step
# sequences, giving useless (1e11+) reconstruction errors. We therefore default to
# the standard, numerically stable ``tanh`` used by virtually all LSTM autoencoders,
# and expose ``relu`` as an opt-in for faithfulness experiments.
DEFAULT_ACTIVATION = "tanh"
GRAD_CLIP_NORM = 5.0

_ACTIVATIONS = {"tanh": nn.tanh, "relu": nn.relu}


@dataclasses.dataclass
class VAEConfig:
    window: int = WINDOW_SIZE
    n_features: int = N_FEATURES
    lstm_units: int = LSTM_UNITS
    latent_dim: int = LATENT_DIM
    activation: str = DEFAULT_ACTIVATION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VAEConfig":
        return cls(window=d["window"], n_features=d["n_features"],
                   lstm_units=d["lstm_units"], latent_dim=d["latent_dim"],
                   activation=d.get("activation", DEFAULT_ACTIVATION))


# ──────────────────────────────────────────────────────────────────────────────
# Flax modules
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    lstm_units: int
    latent_dim: int
    activation: str = DEFAULT_ACTIVATION

    @nn.compact
    def __call__(self, x):                              # x: (B, T, F)
        cell = nn.LSTMCell(features=self.lstm_units,
                           activation_fn=_ACTIVATIONS[self.activation],
                           name="enc_lstm_cell")
        h = nn.RNN(cell, name="enc_rnn")(x)            # (B, T, U)
        pooled = jnp.mean(h, axis=1)                    # global average pooling -> (B, U)
        mu = nn.Dense(self.latent_dim, name="enc_mu")(pooled)
        logvar = nn.Dense(self.latent_dim, name="enc_logvar")(pooled)
        logvar = jnp.clip(logvar, LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class Decoder(nn.Module):
    lstm_units: int
    n_features: int
    window: int
    activation: str = DEFAULT_ACTIVATION

    @nn.compact
    def __call__(self, z):                              # z: (N, L)
        rep = jnp.broadcast_to(z[:, None, :],
                               (z.shape[0], self.window, z.shape[-1]))  # RepeatVector
        cell = nn.LSTMCell(features=self.lstm_units,
                           activation_fn=_ACTIVATIONS[self.activation],
                           name="dec_lstm_cell")
        h = nn.RNN(cell, name="dec_rnn")(rep)          # (N, T, U)
        out = nn.Dense(self.n_features, name="dec_out")(h)  # TimeDistributed Dense
        return out                                      # no output activation (z-scored input)


class VAENet(nn.Module):
    lstm_units: int
    latent_dim: int
    n_features: int
    window: int
    activation: str = DEFAULT_ACTIVATION

    def setup(self):
        self.encoder = Encoder(self.lstm_units, self.latent_dim, self.activation)
        self.decoder = Decoder(self.lstm_units, self.n_features, self.window,
                               self.activation)

    def __call__(self, x, rng, n_samples):              # training forward pass
        mu, logvar = self.encoder(x)
        std = nn.softplus(logvar)
        B = x.shape[0]
        eps = jax.random.normal(rng, (n_samples, B, self.latent_dim))
        z = mu[None] + std[None] * eps                  # (S, B, L)
        recon = self.decoder(z.reshape(n_samples * B, self.latent_dim))
        recon = recon.reshape(n_samples, B, self.window, self.n_features)
        return recon, mu, std

    def encode(self, x):                                # latent mean (deployment: encoder only)
        mu, logvar = self.encoder(x)
        return mu, nn.softplus(logvar)

    def reconstruct(self, x):                           # deterministic reconstruction from mu
        mu, _ = self.encoder(x)
        return self.decoder(mu)


# ──────────────────────────────────────────────────────────────────────────────
# Losses / jitted forward passes
# ──────────────────────────────────────────────────────────────────────────────

def _kl_divergence(mu, std):
    var = std ** 2
    logvar = jnp.log(var + 1e-8)
    kl = -0.5 * jnp.sum(1.0 + logvar - mu ** 2 - var, axis=-1)   # (B,)
    return jnp.mean(kl)


def _loss_fn(params, net, x, rng, beta, n_samples):
    recon, mu, std = net.apply(params, x, rng, n_samples)
    recon_loss = jnp.mean((recon - x[None]) ** 2)       # MC-averaged MSE (Sec. 3.2.2)
    kl_loss = _kl_divergence(mu, std)
    return recon_loss + beta * kl_loss, (recon_loss, kl_loss)


# ──────────────────────────────────────────────────────────────────────────────
# High-level model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class LSTMVAE:
    """
    Trainable LSTM-VAE plus the three anomaly-detection heads from the paper.

    Detection state (built by ``build_detectors``):
        * self.recon_threshold  — max validation reconstruction error × (1+γ)
        * self.dist_threshold   — max validation NN latent distance × (1+γ)
        * self.ball_tree        — BallTree over training latent means (Euclidean)
        * self.iforest          — Isolation Forest over training latent means
    """

    def __init__(self, config: Optional[VAEConfig] = None, params=None, seed: int = 42):
        self.cfg = config or VAEConfig()
        self.net = VAENet(lstm_units=self.cfg.lstm_units,
                          latent_dim=self.cfg.latent_dim,
                          n_features=self.cfg.n_features,
                          window=self.cfg.window,
                          activation=self.cfg.activation)
        key = jax.random.PRNGKey(seed)
        self._key = key
        if params is None:
            dummy = jnp.zeros((1, self.cfg.window, self.cfg.n_features))
            init_key, self._key = jax.random.split(key)
            params = self.net.init(init_key, dummy, jax.random.PRNGKey(0), 1)
        self.params = params

        # Detection artifacts (populated later).
        self.recon_threshold: float = float("inf")
        self.dist_threshold: float = float("inf")
        self.ball_tree: Optional[BallTree] = None
        self.iforest: Optional[IsolationForest] = None
        self.train_means: Optional[np.ndarray] = None

        # Jitted deployment forward passes.
        self._encode = jax.jit(lambda p, x: self.net.apply(p, x, method=self.net.encode))
        self._reconstruct = jax.jit(
            lambda p, x: self.net.apply(p, x, method=self.net.reconstruct))

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, windows: np.ndarray, *, epochs: int = 100, batch_size: int = 1024,
              lr: float = 1e-3, weight_decay: float = 0.0, beta: float = BETA_RECON,
              n_samples: int = N_SAMPLES_TRAIN, seed: int = 42, verbose: bool = True):
        """Train on standardised sequence windows of shape (N, WINDOW, F)."""
        windows = np.asarray(windows, dtype=np.float32)
        n = len(windows)
        steps_per_epoch = max(1, n // batch_size)
        total_steps = max(1, epochs * steps_per_epoch)

        # AdamW + cosine decay to alpha=0 (Table 3.1).
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps,
                                               alpha=0.0)
        optimizer = optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP_NORM),
            optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
        )
        opt_state = optimizer.init(self.params)

        net = self.net

        @partial(jax.jit, static_argnums=(4,))
        def train_step(params, opt_state, x, rng, n_samples):
            (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                params, net, x, rng, beta, n_samples)
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
                b = windows[perm[i:i + batch_size]]
                if len(b) == 0:
                    continue
                key, sub = jax.random.split(key)
                self.params, opt_state, loss = train_step(
                    self.params, opt_state, jnp.asarray(b), sub, n_samples)
                losses.append(float(loss))
            avg = float(np.mean(losses)) if losses else float("nan")
            history.append(avg)
            if verbose:
                print(f"  epoch {ep + 1:3d}/{epochs}  loss={avg:.6f}")
        return history

    # ── batched forward passes ──────────────────────────────────────────────────
    def encode(self, windows: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Return latent means (N, L)."""
        windows = np.asarray(windows, dtype=np.float32)
        out = []
        for i in range(0, len(windows), batch_size):
            mu, _ = self._encode(self.params, jnp.asarray(windows[i:i + batch_size]))
            out.append(np.asarray(mu))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.cfg.latent_dim))

    def reconstruction_error(self, windows: np.ndarray,
                             batch_size: int = 4096) -> np.ndarray:
        """Per-window MSE between input and deterministic reconstruction."""
        windows = np.asarray(windows, dtype=np.float32)
        out = []
        for i in range(0, len(windows), batch_size):
            x = jnp.asarray(windows[i:i + batch_size])
            recon = self._reconstruct(self.params, x)
            err = np.asarray(jnp.mean((recon - x) ** 2, axis=(1, 2)))
            out.append(err)
        return np.concatenate(out, axis=0) if out else np.zeros((0,))

    # ── latent-distance queries (paper Alg. 1/2) ───────────────────────────────
    def latent_distance(self, windows: np.ndarray) -> np.ndarray:
        """Euclidean distance from each window's latent mean to nearest training mean."""
        if self.ball_tree is None:
            raise RuntimeError("Call build_detectors() first.")
        means = self.encode(windows)
        dist, _ = self.ball_tree.query(means, k=1)
        return dist[:, 0]

    # ── latent-clustering scores (paper Alg. 3/4) ──────────────────────────────
    def latent_cluster_score(self, windows: np.ndarray) -> np.ndarray:
        """Isolation-Forest anomaly score (higher = more anomalous)."""
        if self.iforest is None:
            raise RuntimeError("Call build_detectors() first.")
        means = self.encode(windows)
        return -self.iforest.score_samples(means)

    def latent_cluster_predict(self, windows: np.ndarray) -> np.ndarray:
        """Isolation-Forest hard labels: 1 = anomaly, 0 = normal."""
        if self.iforest is None:
            raise RuntimeError("Call build_detectors() first.")
        means = self.encode(windows)
        return (self.iforest.predict(means) == -1).astype(int)

    # ── detector construction / calibration ────────────────────────────────────
    def build_detectors(self, train_windows: np.ndarray, val_windows: np.ndarray,
                        *, margin: float = THRESHOLD_MARGIN,
                        contamination: float = IF_CONTAMINATION):
        """
        Fit the BallTree + Isolation Forest on training latent means and calibrate
        the reconstruction / latent-distance thresholds on a held-out *normal*
        validation set (paper Alg. 1 line 13, and Sec. 3.3.1 threshold rule).
        """
        train_means = self.encode(train_windows)
        self.train_means = train_means

        # Latent distance head — BallTree with Euclidean metric (paper Alg. 1).
        self.ball_tree = BallTree(train_means, metric="euclidean")

        # Latent clustering head — Isolation Forest (paper Alg. 3, Table 3.1).
        self.iforest = IsolationForest(n_estimators=IF_ESTIMATORS,
                                       contamination=contamination,
                                       random_state=IF_RANDOM_STATE, n_jobs=-1)
        self.iforest.fit(train_means)

        # Thresholds from the normal validation set: τ = max(score) · (1 + γ).
        val_recon = self.reconstruction_error(val_windows)
        self.recon_threshold = float(val_recon.max()) * (1.0 + margin)

        val_dist = self.latent_distance(val_windows)
        self.dist_threshold = float(val_dist.max()) * (1.0 + margin)

        return {
            "recon_threshold": self.recon_threshold,
            "dist_threshold": self.dist_threshold,
            "n_train_means": int(len(train_means)),
        }

    # ── persistence ─────────────────────────────────────────────────────────────
    def save(self, directory: str, name: str = "vae"):
        """Serialise weights, config, thresholds, BallTree data and Isolation Forest."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{name}_params.msgpack"), "wb") as f:
            f.write(serialization.to_bytes(self.params))

        meta = {
            "config": self.cfg.to_dict(),
            "recon_threshold": self.recon_threshold,
            "dist_threshold": self.dist_threshold,
        }
        with open(os.path.join(directory, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if self.train_means is not None:
            np.save(os.path.join(directory, f"{name}_train_means.npy"), self.train_means)
        if self.iforest is not None:
            import joblib
            joblib.dump(self.iforest, os.path.join(directory, f"{name}_iforest.joblib"))
        return directory

    @classmethod
    def load(cls, directory: str, name: str = "vae", seed: int = 42) -> "LSTMVAE":
        with open(os.path.join(directory, f"{name}_meta.json")) as f:
            meta = json.load(f)
        cfg = VAEConfig.from_dict(meta["config"])
        model = cls(cfg, seed=seed)

        with open(os.path.join(directory, f"{name}_params.msgpack"), "rb") as f:
            model.params = serialization.from_bytes(model.params, f.read())

        model.recon_threshold = meta["recon_threshold"]
        model.dist_threshold = meta["dist_threshold"]

        means_path = os.path.join(directory, f"{name}_train_means.npy")
        if os.path.exists(means_path):
            model.train_means = np.load(means_path)
            model.ball_tree = BallTree(model.train_means, metric="euclidean")

        if_path = os.path.join(directory, f"{name}_iforest.joblib")
        if os.path.exists(if_path):
            import joblib
            model.iforest = joblib.load(if_path)
        return model
