#!/usr/bin/env python3
"""
fsmn_model.py — FSMN Autoencoder for unsupervised CAN intrusion detection.

Implementation of the lightweight anomaly-detection model proposed in:

    Y. Zhou, J. Zhang, G. Yang, "A Lightweight Unsupervised Intrusion Detection
    Model for In-Vehicle Edge Computing Based on FlexRay", IEEE Access, 2026.

The paper's model (FSMN-AE, Sec. III-A / III-B) is a *non-variational* autoencoder
whose recurrent core is replaced by a Feedforward Sequential Memory Network (FSMN):

    Input      :  (B, T, F) sequence window
    Encoder    :  time-distributed Dense 64 -> 32 -> 16  (ReLU), L1-sparse 16-d code
    Memory     :  FSMN block  h_hat_t = sum_{i=0..N} a_i . h_{t-i}   (Eq. 1, N=10)
                  fused_t = ReLU( W h_t + W_hat h_hat_t )            (Eq. 2)
    Decoder    :  time-distributed Dense 32 -> 64 -> F  (tanh hidden, linear out)
    Loss       :  MSE reconstruction  +  lambda * L1(code)           (Eq. 3 + sparsity)
    Detection  :  per-window reconstruction MSE vs dynamic threshold
                  theta = mu_train + k * sigma_train                 (Eq. 4, k=2.5)

Why FSMN instead of an RNN (the paper's core claim): the memory block is a *linear*
weighted sum over a fixed window of N past hidden states, so it drops the recurrent
state chain. This cuts time complexity from O(T.D^2) (RNN) to O(T.D.N) and shrinks the
parameter count dramatically (paper reports ~91% fewer parameters than a GRU-AE),
which is the whole point for resource-constrained in-vehicle edge deployment.

ADAPTATION TO TRANSLATED CAN DATA
---------------------------------
The paper targets *raw FlexRay frames* (ID / payload / CRC) with a memory window rigidly
matched to the FlexRay static-segment cycle (10 frames x 5 ms). Two things change here,
and only these — the detector itself is unchanged:

  1. Protocol-specific framing is dropped. This repository's data is *already translated*
     CAN — decoded continuous signals (speed, battery, throttle, brake, steering, gear,
     location_x, location_y). There is no static/dynamic-segment split, no CRC, no
     arbitration-ID field, so the FlexRay dual-mode (static vs dynamic) feature logic is
     not applicable; the signals are fed directly into the encoder, z-score standardised.

  2. The memory order N is re-interpreted for CAN. In the paper N=10 covers 10 FlexRay
     static cycles; here N is simply the number of past message-steps the memory block
     mixes, and the sequence window (WINDOW_SIZE / STRIDE) matches the sibling vae_v1 /
     mvp_v1 models so results are directly comparable. N defaults to 10 (paper value) but
     is a free hyperparameter you can retune to the CAN message periodicity.

Everything else — the unsupervised reconstruction objective, the L1-sparse latent, the
mu + k.sigma dynamic threshold — is faithful to the paper.
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


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (paper Sec. III-A / III-B, Table 2)
# ──────────────────────────────────────────────────────────────────────────────

# Sequence window over decoded-CAN messages. Kept identical to vae_v1 / mvp_v1 so the
# three models are compared on the same inputs. (In the FlexRay paper the "window" is
# the FSMN memory order below, not this sequence length.)
WINDOW_SIZE = 50
STRIDE = 20

# Decoded-CAN signal channels used as model input.
FEATURES = ("speed_kmh", "battery_level", "throttle", "brake",
            "steering", "gear", "location_x", "location_y")
N_FEATURES = len(FEATURES)

# Encoder compression path (paper: gradient dimension reduction 64 -> 32 -> 16).
ENC_DIMS = (64, 32, 16)
CODE_DIM = ENC_DIMS[-1]          # = FSMN hidden width (paper keeps these consistent, 16-d)

# FSMN memory block (paper Eq. 1, Sec. III-A): 1 x 10 frame window.
FSMN_ORDER = 10                  # N backward taps  (h_{t-1} .. h_{t-N})
FSMN_ORDER_FWD = 0               # forward (look-ahead) taps; >0 => bidirectional memory

# L1 sparsity penalty on the latent code (paper: lambda = 0.001, ~85% sparsity).
L1_LAMBDA = 1e-3
DROPOUT_RATE = 0.1               # paper uses Dropout + L1 dual regularisation

# Dynamic anomaly threshold  theta = mu_train + k * sigma_train  (paper Eq. 4).
THRESHOLD_K = 2.5                # optimal k from the paper's sensitivity sweep (Table 3)

# Optimisation.
DEFAULT_ACTIVATION = "relu"      # encoder activation; paper uses ReLU (stable here: no
                                 # unbounded recurrent state, unlike the vae_v1 LSTM)
GRAD_CLIP_NORM = 5.0

_ACTIVATIONS = {"relu": nn.relu, "tanh": nn.tanh}


@dataclasses.dataclass
class FSMNAEConfig:
    window: int = WINDOW_SIZE
    n_features: int = N_FEATURES
    enc_dims: tuple = ENC_DIMS
    code_dim: int = CODE_DIM
    fsmn_order: int = FSMN_ORDER
    fsmn_order_fwd: int = FSMN_ORDER_FWD
    dropout: float = DROPOUT_RATE
    activation: str = DEFAULT_ACTIVATION

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["enc_dims"] = list(self.enc_dims)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FSMNAEConfig":
        return cls(
            window=d["window"],
            n_features=d["n_features"],
            enc_dims=tuple(d.get("enc_dims", ENC_DIMS)),
            code_dim=d.get("code_dim", d.get("enc_dims", ENC_DIMS)[-1]),
            fsmn_order=d.get("fsmn_order", FSMN_ORDER),
            fsmn_order_fwd=d.get("fsmn_order_fwd", FSMN_ORDER_FWD),
            dropout=d.get("dropout", DROPOUT_RATE),
            activation=d.get("activation", DEFAULT_ACTIVATION),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Flax modules
# ──────────────────────────────────────────────────────────────────────────────

class FSMNMemory(nn.Module):
    """
    Vectorised Feedforward Sequential Memory block (paper Eq. 1).

        h_hat_t = a_0 . h_t + sum_{i=1..N} a_i^back . h_{t-i}
                            + sum_{j=1..M} a_j^fwd  . h_{t+j}

    The per-tap, per-dimension coefficients ``a`` are learned by global optimisation
    (no recurrent state). Backward taps look at history; optional forward taps give the
    bidirectional "past + future" memory the paper describes (Sec. III-A).
    """
    order: int
    order_fwd: int
    features: int

    @nn.compact
    def __call__(self, h):                                  # h: (B, T, D)
        B, T, D = h.shape
        a_back = self.param("a_back", nn.initializers.normal(0.1),
                            (self.order + 1, self.features))
        mem = a_back[0] * h                                 # i = 0 (current frame)
        for i in range(1, self.order + 1):                  # backward taps h_{t-i}
            shifted = jnp.concatenate(
                [jnp.zeros((B, i, D), h.dtype), h[:, :T - i, :]], axis=1)
            mem = mem + a_back[i] * shifted
        if self.order_fwd > 0:                              # forward taps h_{t+j}
            a_fwd = self.param("a_fwd", nn.initializers.normal(0.1),
                              (self.order_fwd, self.features))
            for j in range(1, self.order_fwd + 1):
                shifted = jnp.concatenate(
                    [h[:, j:, :], jnp.zeros((B, j, D), h.dtype)], axis=1)
                mem = mem + a_fwd[j - 1] * shifted
        return mem                                          # h_hat: (B, T, D)


class Encoder(nn.Module):
    enc_dims: tuple
    activation: str = DEFAULT_ACTIVATION

    @nn.compact
    def __call__(self, x):                                  # x: (B, T, F)
        act = _ACTIVATIONS[self.activation]
        h = x
        for k, dim in enumerate(self.enc_dims):
            h = nn.Dense(dim, name=f"enc_dense_{k}")(h)      # time-distributed
            h = act(h)
        return h                                            # sparse code (B, T, code_dim)


class Decoder(nn.Module):
    dec_dims: tuple
    n_features: int

    @nn.compact
    def __call__(self, h):                                  # h: (B, T, code_dim)
        for k, dim in enumerate(self.dec_dims):
            h = nn.Dense(dim, name=f"dec_dense_{k}")(h)
            h = nn.tanh(h)                                   # bounded decoder (paper: tanh)
        out = nn.Dense(self.n_features, name="dec_out")(h)   # linear output (z-scored input)
        return out


class FSMNAENet(nn.Module):
    enc_dims: tuple
    code_dim: int
    n_features: int
    window: int
    fsmn_order: int
    fsmn_order_fwd: int
    dropout: float
    activation: str = DEFAULT_ACTIVATION

    def setup(self):
        self.encoder = Encoder(self.enc_dims, self.activation)
        self.memory = FSMNMemory(self.fsmn_order, self.fsmn_order_fwd, self.code_dim)
        # Eq. (2): fused_t = ReLU(W h_t + W_hat h_hat_t)
        self.w_h = nn.Dense(self.code_dim, name="fuse_h")
        self.w_mem = nn.Dense(self.code_dim, name="fuse_mem")
        self.drop = nn.Dropout(rate=self.dropout)
        # Mirror of the encoder path for the decoder (16 -> 32 -> 64 -> F).
        self.decoder = Decoder(tuple(reversed(self.enc_dims[:-1])), self.n_features)

    def __call__(self, x, deterministic: bool = True):      # training/inference forward
        code = self.encoder(x)                              # (B, T, code_dim)
        h_hat = self.memory(code)
        fused = _ACTIVATIONS[self.activation](self.w_h(code) + self.w_mem(h_hat))
        fused = self.drop(fused, deterministic=deterministic)
        recon = self.decoder(fused)
        return recon, code

    def reconstruct(self, x):                               # deterministic reconstruction
        recon, _ = self(x, deterministic=True)
        return recon

    def latent_code(self, x):                               # sparse code (analysis only)
        return self.encoder(x)


# ──────────────────────────────────────────────────────────────────────────────
# Loss / jitted forward passes
# ──────────────────────────────────────────────────────────────────────────────

def _loss_fn(params, net, x, dropout_rng, l1_lambda):
    recon, code = net.apply(params, x, deterministic=False, rngs={"dropout": dropout_rng})
    recon_loss = jnp.mean((recon - x) ** 2)                 # MSE reconstruction (Eq. 3)
    l1_loss = jnp.mean(jnp.abs(code))                       # sparsity penalty on latent
    return recon_loss + l1_lambda * l1_loss, (recon_loss, l1_loss)


# ──────────────────────────────────────────────────────────────────────────────
# High-level model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class FSMNAE:
    """
    Trainable FSMN autoencoder plus the paper's reconstruction-error detector.

    Detection state (built by ``build_detector``):
        * self.recon_threshold  — theta = mu_train + k * sigma_train  (paper Eq. 4)
        * self.err_mean / self.err_std / self.k  — the components of that threshold
    """

    def __init__(self, config: Optional[FSMNAEConfig] = None, params=None, seed: int = 42):
        self.cfg = config or FSMNAEConfig()
        self.net = FSMNAENet(
            enc_dims=self.cfg.enc_dims,
            code_dim=self.cfg.code_dim,
            n_features=self.cfg.n_features,
            window=self.cfg.window,
            fsmn_order=self.cfg.fsmn_order,
            fsmn_order_fwd=self.cfg.fsmn_order_fwd,
            dropout=self.cfg.dropout,
            activation=self.cfg.activation,
        )
        key = jax.random.PRNGKey(seed)
        self._key = key
        if params is None:
            dummy = jnp.zeros((1, self.cfg.window, self.cfg.n_features))
            init_key, self._key = jax.random.split(key)
            params = self.net.init({"params": init_key, "dropout": jax.random.PRNGKey(0)},
                                   dummy, deterministic=True)
        self.params = params

        # Detection artifacts (populated by build_detector).
        self.recon_threshold: float = float("inf")
        self.err_mean: float = 0.0
        self.err_std: float = 0.0
        self.k: float = THRESHOLD_K

        # Jitted deterministic reconstruction for deployment.
        self._reconstruct = jax.jit(
            lambda p, x: self.net.apply(p, x, method=self.net.reconstruct))

    # ── info ────────────────────────────────────────────────────────────────────
    def num_params(self) -> int:
        """Total learnable scalars — the paper's headline lightweight metric."""
        return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(self.params)))

    def approx_size_kb(self) -> float:
        """Approximate float32 model footprint in KiB."""
        return self.num_params() * 4 / 1024.0

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, windows: np.ndarray, *, epochs: int = 100, batch_size: int = 1024,
              lr: float = 1e-3, weight_decay: float = 0.0, l1_lambda: float = L1_LAMBDA,
              seed: int = 42, verbose: bool = True):
        """Train on standardised sequence windows of shape (N, WINDOW, F)."""
        windows = np.asarray(windows, dtype=np.float32)
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
        def train_step(params, opt_state, x, dropout_rng):
            (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                params, net, x, dropout_rng, l1_lambda)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, aux

        rng = np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed)
        history = []
        for ep in range(epochs):
            perm = rng.permutation(n)
            losses, recons, l1s = [], [], []
            for i in range(0, n, batch_size):
                b = windows[perm[i:i + batch_size]]
                if len(b) == 0:
                    continue
                key, sub = jax.random.split(key)
                self.params, opt_state, loss, aux = train_step(
                    self.params, opt_state, jnp.asarray(b), sub)
                losses.append(float(loss)); recons.append(float(aux[0])); l1s.append(float(aux[1]))
            avg = float(np.mean(losses)) if losses else float("nan")
            history.append(avg)
            if verbose:
                print(f"  epoch {ep + 1:3d}/{epochs}  loss={avg:.6f}  "
                      f"(recon={np.mean(recons):.6f}  l1={np.mean(l1s):.6f})")
        return history

    # ── forward passes ──────────────────────────────────────────────────────────
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

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Hard labels: 1 = anomaly (error > dynamic threshold), 0 = normal."""
        return (self.reconstruction_error(windows) > self.recon_threshold).astype(int)

    # ── detector calibration (paper Eq. 4) ──────────────────────────────────────
    def build_detector(self, train_windows: np.ndarray, *, k: float = THRESHOLD_K):
        """
        Calibrate the dynamic threshold theta = mu + k*sigma on the reconstruction-error
        distribution of *normal* training data (paper Sec. III-C, Eq. 4).
        """
        errs = self.reconstruction_error(train_windows)
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

    # ── persistence ─────────────────────────────────────────────────────────────
    def save(self, directory: str, name: str = "fsmn_ae"):
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
    def load(cls, directory: str, name: str = "fsmn_ae", seed: int = 42) -> "FSMNAE":
        with open(os.path.join(directory, f"{name}_meta.json")) as f:
            meta = json.load(f)
        cfg = FSMNAEConfig.from_dict(meta["config"])
        model = cls(cfg, seed=seed)
        with open(os.path.join(directory, f"{name}_params.msgpack"), "rb") as f:
            model.params = serialization.from_bytes(model.params, f.read())
        model.recon_threshold = meta["recon_threshold"]
        model.err_mean = meta.get("err_mean", 0.0)
        model.err_std = meta.get("err_std", 0.0)
        model.k = meta.get("k", THRESHOLD_K)
        return model
