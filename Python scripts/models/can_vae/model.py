#!/usr/bin/env python3
"""
can_vae — LSTM β-VAE for *raw* CAN frames with Chowdhury ID embedding.

Same three anomaly heads as ``models.vae.LSTMVAE`` (Chowdhury 2025):
  1. reconstruction error
  2. latent nearest-neighbour distance (BallTree)
  3. latent Isolation Forest

Input windows are engineered raw-CAN features ``(N, T, N_RAW_FEATURES)``;
the first ``ID_BITS`` columns are projected by a learnable Dense embedding
before the LSTM encoder/decoder (same FE path as ``can_gru``).
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

from models.can_gru.features import (
    EMBED_DIM,
    FEATURES,
    ID_BITS,
    N_MODEL_FEATURES,
    N_RAW_FEATURES,
)
from models.vae.model import (
    BETA_LATENT,
    BETA_RECON,
    DEFAULT_ACTIVATION,
    GRAD_CLIP_NORM,
    IF_CONTAMINATION,
    IF_ESTIMATORS,
    IF_RANDOM_STATE,
    LATENT_DIM,
    LSTM_UNITS,
    N_SAMPLES_TRAIN,
    THRESHOLD_MARGIN,
    Decoder,
    Encoder,
    _kl_divergence,
)

# Paper-aligned CAN window (same as can_gru).
WINDOW_SIZE = 50
STRIDE = 50


@dataclasses.dataclass
class CANVAEConfig:
    window: int = WINDOW_SIZE
    n_raw_features: int = N_RAW_FEATURES
    n_model_features: int = N_MODEL_FEATURES
    embed_dim: int = EMBED_DIM
    id_bits: int = ID_BITS
    lstm_units: int = LSTM_UNITS
    latent_dim: int = LATENT_DIM
    activation: str = DEFAULT_ACTIVATION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CANVAEConfig":
        return cls(
            window=int(d.get("window", WINDOW_SIZE)),
            n_raw_features=int(d.get("n_raw_features", N_RAW_FEATURES)),
            n_model_features=int(d.get("n_model_features", N_MODEL_FEATURES)),
            embed_dim=int(d.get("embed_dim", EMBED_DIM)),
            id_bits=int(d.get("id_bits", ID_BITS)),
            lstm_units=int(d.get("lstm_units", LSTM_UNITS)),
            latent_dim=int(d.get("latent_dim", LATENT_DIM)),
            activation=d.get("activation", DEFAULT_ACTIVATION),
        )


class IdEmbed(nn.Module):
    """Learnable projection of binary arbitration-ID bits → continuous embedding."""

    embed_dim: int = EMBED_DIM
    id_bits: int = ID_BITS

    @nn.compact
    def __call__(self, x):  # x: (B, T, N_RAW)
        bits = x[..., : self.id_bits]
        rest = x[..., self.id_bits :]
        emb = nn.Dense(self.embed_dim, name="id_embed")(bits)
        return jnp.concatenate([emb, rest], axis=-1)


class CANVAENet(nn.Module):
    lstm_units: int
    latent_dim: int
    n_model_features: int
    window: int
    embed_dim: int = EMBED_DIM
    id_bits: int = ID_BITS
    activation: str = DEFAULT_ACTIVATION

    def setup(self):
        self.embed = IdEmbed(embed_dim=self.embed_dim, id_bits=self.id_bits)
        self.encoder = Encoder(self.lstm_units, self.latent_dim, self.activation)
        self.decoder = Decoder(
            self.lstm_units, self.n_model_features, self.window, self.activation
        )

    def __call__(self, x, rng, n_samples):
        x_e = self.embed(x)
        mu, logvar = self.encoder(x_e)
        std = nn.softplus(logvar)
        B = x.shape[0]
        eps = jax.random.normal(rng, (n_samples, B, self.latent_dim))
        z = mu[None] + std[None] * eps
        recon = self.decoder(z.reshape(n_samples * B, self.latent_dim))
        recon = recon.reshape(n_samples, B, self.window, self.n_model_features)
        return recon, mu, std, x_e

    def encode(self, x):
        x_e = self.embed(x)
        mu, logvar = self.encoder(x_e)
        return mu, nn.softplus(logvar)

    def reconstruct(self, x):
        x_e = self.embed(x)
        mu, _ = self.encoder(x_e)
        return self.decoder(mu), x_e


def _loss_fn(params, net, x, rng, beta, n_samples):
    recon, mu, std, x_e = net.apply(params, x, rng, n_samples)
    recon_loss = jnp.mean((recon - x_e[None]) ** 2)
    kl_loss = _kl_divergence(mu, std)
    return recon_loss + beta * kl_loss, (recon_loss, kl_loss)


class CANLSTMVAE:
    """
    Trainable CAN LSTM-VAE + three anomaly-detection heads.

    Detection state (``build_detectors``):
        * recon_threshold
        * dist_threshold
        * ball_tree
        * iforest
    """

    features = FEATURES

    def __init__(
        self,
        config: Optional[CANVAEConfig] = None,
        params=None,
        seed: int = 42,
    ):
        self.cfg = config or CANVAEConfig()
        if self.cfg.n_raw_features != N_RAW_FEATURES:
            raise ValueError(
                f"can_vae expects n_raw_features={N_RAW_FEATURES}, "
                f"got {self.cfg.n_raw_features}"
            )
        self.net = CANVAENet(
            lstm_units=self.cfg.lstm_units,
            latent_dim=self.cfg.latent_dim,
            n_model_features=self.cfg.n_model_features,
            window=self.cfg.window,
            embed_dim=self.cfg.embed_dim,
            id_bits=self.cfg.id_bits,
            activation=self.cfg.activation,
        )
        key = jax.random.PRNGKey(seed)
        self._key = key
        if params is None:
            dummy = jnp.zeros((1, self.cfg.window, self.cfg.n_raw_features))
            init_key, self._key = jax.random.split(key)
            params = self.net.init(init_key, dummy, jax.random.PRNGKey(0), 1)
        self.params = params

        self.recon_threshold: float = float("inf")
        self.dist_threshold: float = float("inf")
        self.ball_tree: Optional[BallTree] = None
        self.iforest: Optional[IsolationForest] = None
        self.train_means: Optional[np.ndarray] = None

        self._encode = jax.jit(
            lambda p, x: self.net.apply(p, x, method=self.net.encode)
        )
        self._reconstruct = jax.jit(
            lambda p, x: self.net.apply(p, x, method=self.net.reconstruct)
        )

    def train(
        self,
        windows: np.ndarray,
        *,
        epochs: int = 100,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        beta: float = BETA_RECON,
        n_samples: int = N_SAMPLES_TRAIN,
        seed: int = 42,
        verbose: bool = True,
    ):
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 3 or windows.shape[-1] != self.cfg.n_raw_features:
            raise ValueError(
                f"Expected windows (N, T, {self.cfg.n_raw_features}), got {windows.shape}"
            )
        n = len(windows)
        steps_per_epoch = max(1, n // batch_size)
        total_steps = max(1, epochs * steps_per_epoch)

        schedule = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=total_steps, alpha=0.0
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP_NORM),
            optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
        )
        opt_state = optimizer.init(self.params)
        net = self.net

        @partial(jax.jit, static_argnums=(4,))
        def train_step(params, opt_state, x, rng, n_samples_):
            (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                params, net, x, rng, beta, n_samples_
            )
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
                b = windows[perm[i : i + batch_size]]
                if len(b) == 0:
                    continue
                key, sub = jax.random.split(key)
                self.params, opt_state, loss = train_step(
                    self.params, opt_state, jnp.asarray(b), sub, n_samples
                )
                losses.append(float(loss))
            avg = float(np.mean(losses)) if losses else float("nan")
            history.append(avg)
            if verbose:
                print(f"  epoch {ep + 1:3d}/{epochs}  loss={avg:.6f}")
        return history

    def encode(self, windows: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float32)
        out = []
        for i in range(0, len(windows), batch_size):
            mu, _ = self._encode(self.params, jnp.asarray(windows[i : i + batch_size]))
            out.append(np.asarray(mu))
        return (
            np.concatenate(out, axis=0)
            if out
            else np.zeros((0, self.cfg.latent_dim), dtype=np.float32)
        )

    def reconstruction_error(
        self, windows: np.ndarray, batch_size: int = 4096
    ) -> np.ndarray:
        """Per-window MSE in embedded feature space (deterministic μ decode)."""
        windows = np.asarray(windows, dtype=np.float32)
        out = []
        for i in range(0, len(windows), batch_size):
            x = jnp.asarray(windows[i : i + batch_size])
            recon, x_e = self._reconstruct(self.params, x)
            err = np.asarray(jnp.mean((recon - x_e) ** 2, axis=(1, 2)))
            out.append(err)
        return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)

    def latent_distance(self, windows: np.ndarray) -> np.ndarray:
        if self.ball_tree is None:
            raise RuntimeError("Call build_detectors() first.")
        means = self.encode(windows)
        dist, _ = self.ball_tree.query(means, k=1)
        return dist[:, 0]

    def latent_cluster_score(self, windows: np.ndarray) -> np.ndarray:
        if self.iforest is None:
            raise RuntimeError("Call build_detectors() first.")
        means = self.encode(windows)
        return -self.iforest.score_samples(means)

    def latent_cluster_predict(self, windows: np.ndarray) -> np.ndarray:
        if self.iforest is None:
            raise RuntimeError("Call build_detectors() first.")
        means = self.encode(windows)
        return (self.iforest.predict(means) == -1).astype(int)

    def build_detectors(
        self,
        train_windows: np.ndarray,
        val_windows: np.ndarray,
        *,
        margin: float = THRESHOLD_MARGIN,
        contamination: float = IF_CONTAMINATION,
    ):
        train_means = self.encode(train_windows)
        self.train_means = train_means
        self.ball_tree = BallTree(train_means, metric="euclidean")
        self.iforest = IsolationForest(
            n_estimators=IF_ESTIMATORS,
            contamination=contamination,
            random_state=IF_RANDOM_STATE,
            n_jobs=-1,
        )
        self.iforest.fit(train_means)

        val_recon = self.reconstruction_error(val_windows)
        self.recon_threshold = float(val_recon.max()) * (1.0 + margin)
        val_dist = self.latent_distance(val_windows)
        self.dist_threshold = float(val_dist.max()) * (1.0 + margin)
        return {
            "recon_threshold": self.recon_threshold,
            "dist_threshold": self.dist_threshold,
            "n_train_means": int(len(train_means)),
        }

    def num_params(self) -> int:
        return int(
            sum(np.prod(np.array(v).shape) for v in jax.tree_util.tree_leaves(self.params))
        )

    def save(self, directory: str, name: str = "can_vae"):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{name}_params.msgpack"), "wb") as f:
            f.write(serialization.to_bytes(self.params))
        # Use *_model_meta.json so ids_core.trainer's {name}_meta.json does not
        # overwrite recon/dist thresholds + Flax config.
        meta = {
            "config": self.cfg.to_dict(),
            "recon_threshold": self.recon_threshold,
            "dist_threshold": self.dist_threshold,
            "features": list(FEATURES),
        }
        with open(os.path.join(directory, f"{name}_model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if self.train_means is not None:
            np.save(
                os.path.join(directory, f"{name}_train_means.npy"), self.train_means
            )
        if self.iforest is not None:
            import joblib

            joblib.dump(self.iforest, os.path.join(directory, f"{name}_iforest.joblib"))
        return directory

    @classmethod
    def load(cls, directory: str, name: str = "can_vae", seed: int = 42) -> "CANLSTMVAE":
        model_meta = os.path.join(directory, f"{name}_model_meta.json")
        legacy_meta = os.path.join(directory, f"{name}_meta.json")
        meta_path = model_meta if os.path.exists(model_meta) else legacy_meta
        with open(meta_path) as f:
            meta = json.load(f)
        if "config" not in meta:
            raise FileNotFoundError(
                f"Expected model meta with 'config' at {model_meta} "
                f"(trainer-only meta is not enough). Retrain can_vae."
            )
        cfg = CANVAEConfig.from_dict(meta["config"])
        model = cls(cfg, seed=seed)
        with open(os.path.join(directory, f"{name}_params.msgpack"), "rb") as f:
            model.params = serialization.from_bytes(model.params, f.read())
        model.recon_threshold = float(meta["recon_threshold"])
        model.dist_threshold = float(meta["dist_threshold"])
        means_path = os.path.join(directory, f"{name}_train_means.npy")
        if os.path.exists(means_path):
            model.train_means = np.load(means_path)
            model.ball_tree = BallTree(model.train_means, metric="euclidean")
        if_path = os.path.join(directory, f"{name}_iforest.joblib")
        if os.path.exists(if_path):
            import joblib

            model.iforest = joblib.load(if_path)
        return model
