"""VAE adapter."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from models.vae import (
    BETA_LATENT,
    BETA_RECON,
    IF_CONTAMINATION,
    LSTMVAE,
    N_SAMPLES_TRAIN,
    THRESHOLD_MARGIN,
    VAEConfig,
)

from .base import FitResult, ModelSpec

SPEC = ModelSpec(
    key="vae",
    display_name="VAE",
    default_ckpt_name="vae",
    default_epochs=50,
    default_batch_size=256,
)


class VAEAdapter:
    spec = SPEC

    def __init__(
        self,
        *,
        n_features: int,
        window_size: int,
        seed: int = 42,
        objective: str = "recon",
        lr: float = 1e-3,
        contamination: float = IF_CONTAMINATION,
        margin: float = THRESHOLD_MARGIN,
        **_: Any,
    ):
        self._lr = lr
        self._contamination = contamination
        self._margin = margin
        self._beta = BETA_RECON if objective == "recon" else BETA_LATENT
        self._model = LSTMVAE(VAEConfig(window=window_size, n_features=n_features), seed=seed)

    def fit(
        self,
        train_windows: np.ndarray,
        *,
        val_windows: Optional[np.ndarray] = None,
        train_timestamps: Optional[np.ndarray] = None,
        val_timestamps: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 256,
        seed: int = 42,
        **kwargs: Any,
    ) -> FitResult:
        _ = train_timestamps, val_timestamps
        lr = float(kwargs.get("lr", self._lr))
        beta = float(kwargs.get("beta", self._beta))
        margin = float(kwargs.get("margin", self._margin))
        contamination = float(kwargs.get("contamination", self._contamination))
        t0 = time.perf_counter()
        self._model.train(
            train_windows, epochs=epochs, batch_size=batch_size, lr=lr,
            beta=beta, n_samples=N_SAMPLES_TRAIN, seed=seed, verbose=True,
        )
        print(f"[Train] done in {time.perf_counter() - t0:.1f}s")
        cal_w = val_windows if val_windows is not None and len(val_windows) else train_windows
        info = self._model.build_detectors(
            train_windows, cal_w, margin=margin, contamination=contamination,
        )
        thr = float(self._model.recon_threshold)
        print(f"[Calibrate] recon_threshold={thr:.6f}")
        return FitResult(threshold=thr, extras=dict(info) if isinstance(info, dict) else {})

    def score(self, windows: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        _ = timestamps
        return np.asarray(self._model.reconstruction_error(windows), dtype=np.float64)

    @property
    def threshold(self) -> float:
        return float(self._model.recon_threshold)

    def save(self, outdir: Path, name: str) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        self._model.save(str(outdir), name=name)

    def num_params(self) -> int:
        if hasattr(self._model, "num_params"):
            return int(self._model.num_params())
        import jax
        return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(self._model.params)))
