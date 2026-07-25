"""can_vae adapter — raw-CAN LSTM β-VAE with three anomaly heads."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from models.can_gru.features import N_RAW_FEATURES
from models.can_vae import (
    BETA_LATENT,
    BETA_RECON,
    CANLSTMVAE,
    CANVAEConfig,
    IF_CONTAMINATION,
    N_SAMPLES_TRAIN,
    THRESHOLD_MARGIN,
    WINDOW_SIZE,
)

from .base import FitResult, ModelSpec

SPEC = ModelSpec(
    key="can_vae",
    display_name="CAN-VAE",
    default_ckpt_name="can_vae",
    needs_timestamps=False,
    uses_flat_windows=False,
    default_epochs=50,
    default_batch_size=256,
)


class CANVAEAdapter:
    """Separate from telemetry ``vae`` — expects engineered raw-CAN features."""

    spec = SPEC

    def __init__(
        self,
        *,
        n_features: int,
        window_size: int = WINDOW_SIZE,
        seed: int = 42,
        objective: str = "recon",
        lr: float = 1e-3,
        contamination: float = IF_CONTAMINATION,
        margin: float = THRESHOLD_MARGIN,
        **_: Any,
    ):
        if n_features != N_RAW_FEATURES:
            raise ValueError(
                f"can_vae requires the 'can' profile with {N_RAW_FEATURES} features, "
                f"got n_features={n_features}. Preprocess with: "
                f"python -m datasets.can.preprocess"
            )
        self._lr = lr
        self._contamination = contamination
        self._margin = margin
        self._beta = BETA_RECON if objective == "recon" else BETA_LATENT
        self._model = CANLSTMVAE(
            CANVAEConfig(window=window_size, n_raw_features=n_features),
            seed=seed,
        )

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
        history = self._model.train(
            train_windows,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            beta=beta,
            n_samples=N_SAMPLES_TRAIN,
            seed=seed,
            verbose=True,
        )
        print(f"[Train] done in {time.perf_counter() - t0:.1f}s")
        cal_w = (
            val_windows
            if val_windows is not None and len(val_windows)
            else train_windows
        )
        info = self._model.build_detectors(
            train_windows, cal_w, margin=margin, contamination=contamination
        )
        thr = float(self._model.recon_threshold)
        print(f"[Calibrate] recon_threshold={thr:.6f}")
        print(f"[Calibrate] dist_threshold={self._model.dist_threshold:.6f}")
        last_loss = float(history[-1]) if history else None
        return FitResult(
            threshold=thr,
            train_loss=last_loss,
            extras=dict(info) if isinstance(info, dict) else {},
        )

    def score(
        self, windows: np.ndarray, timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        _ = timestamps
        return np.asarray(self._model.reconstruction_error(windows), dtype=np.float64)

    @property
    def threshold(self) -> float:
        return float(self._model.recon_threshold)

    def save(self, outdir: Path, name: str) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self._model.save(str(outdir), name=name)

    def num_params(self) -> int:
        return int(self._model.num_params())
