"""TET adapter."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from models.tet import (
    AE_HIDDEN,
    AE_LATENT,
    D_MODEL,
    DROPOUT_RATE,
    FF_DIM,
    N_HEADS,
    N_LAYERS,
    NOISE_STD,
    TETAE,
    TETConfig,
    THRESHOLD_K,
)

from .base import FitResult, ModelSpec

SPEC = ModelSpec(
    key="tet",
    display_name="TET",
    default_ckpt_name="tet",
    needs_timestamps=True,
    default_epochs=50,
    default_batch_size=256,
)


class TETAdapter:
    spec = SPEC

    def __init__(
        self,
        *,
        n_features: int,
        window_size: int,
        seed: int = 42,
        ae_latent: int | None = None,
        ae_hidden: int = AE_HIDDEN,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT_RATE,
        noise_std: float = NOISE_STD,
        k: float = THRESHOLD_K,
        lr: float = 1e-3,
        **_: Any,
    ):
        self._k, self._lr, self._noise_std = k, lr, noise_std
        latent = ae_latent if ae_latent is not None else max(AE_LATENT, n_features // 2)
        cfg = TETConfig(
            window=window_size,
            n_features=n_features,
            ae_latent=latent,
            ae_hidden=ae_hidden,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            noise_std=noise_std,
        )
        self._model = TETAE(cfg, seed=seed)

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
        if train_timestamps is None:
            raise ValueError("TETAdapter.fit requires train_timestamps")
        lr = float(kwargs.get("lr", self._lr))
        noise_std = float(kwargs.get("noise_std", self._noise_std))
        k = float(kwargs.get("k", self._k))
        print(f"[Model] params={self._model.num_params():,}  (~{self._model.approx_size_kb():.1f} KiB)")
        t0 = time.perf_counter()
        self._model.train(
            train_windows, train_timestamps,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=0.0,
            noise_std=noise_std, seed=seed, verbose=True,
        )
        print(f"[Train] done in {time.perf_counter() - t0:.1f}s")
        info = self._model.build_detector(train_windows, train_timestamps, k=k)
        print(f"[Calibrate] theta={info['recon_threshold']:.6f}")
        val_fpr = None
        if val_windows is not None and len(val_windows) and val_timestamps is not None:
            val_err = self._model.reconstruction_error(val_windows, val_timestamps)
            val_fpr = float((val_err > info["recon_threshold"]).mean())
            print(f"[Calibrate] held-out normal FPR={val_fpr:.4f}")
        return FitResult(threshold=float(info["recon_threshold"]), val_fpr=val_fpr, extras=info)

    def score(self, windows: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        if timestamps is None:
            raise ValueError("TETAdapter.score requires timestamps")
        return np.asarray(self._model.reconstruction_error(windows, timestamps), dtype=np.float64)

    @property
    def threshold(self) -> float:
        return float(self._model.recon_threshold)

    def save(self, outdir: Path, name: str) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        self._model.save(str(outdir), name=name)

    def num_params(self) -> int:
        return int(self._model.num_params())
