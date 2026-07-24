"""can_gru adapter — raw-CAN GRU-AE with Chowdhury feature engineering."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from models.can_gru import CANGRUAutoencoder
from models.can_gru.features import N_RAW_FEATURES
from models.can_gru.model import GRU_HIDDEN, WINDOW_SIZE

from ..data import flatten_windows
from .base import FitResult, ModelSpec

SPEC = ModelSpec(
    key="can_gru",
    display_name="CAN-GRU-AE",
    default_ckpt_name="can_gru",
    needs_timestamps=False,
    uses_flat_windows=True,
    default_epochs=20,
    default_batch_size=64,
)


class CANGRUAdapter:
    """Separate from telemetry ``gru`` — expects engineered raw-CAN features."""

    spec = SPEC

    def __init__(
        self,
        *,
        n_features: int,
        window_size: int = WINDOW_SIZE,
        seed: int = 42,
        checkpoint_dir: Path | None = None,
        safety_mult: float = 1.0,
        hidden: int = GRU_HIDDEN,
        **_: Any,
    ):
        if n_features != N_RAW_FEATURES:
            raise ValueError(
                f"can_gru requires the 'can' profile with {N_RAW_FEATURES} features, "
                f"got n_features={n_features}. Preprocess with: "
                f"python -m datasets.can.preprocess"
            )
        self._safety_mult = safety_mult
        self._model = CANGRUAutoencoder(
            n_features=n_features,
            window_size=window_size,
            hidden=hidden,
            seed=seed,
            checkpoint_dir=checkpoint_dir or Path("checkpoints"),
        )

    def fit(
        self,
        train_windows: np.ndarray,
        *,
        val_windows: Optional[np.ndarray] = None,
        train_timestamps: Optional[np.ndarray] = None,
        val_timestamps: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 64,
        seed: int = 42,
        **kwargs: Any,
    ) -> FitResult:
        _ = train_timestamps, val_timestamps, kwargs
        train_flat = self._as_flat(train_windows)
        val_flat = (
            self._as_flat(val_windows)
            if val_windows is not None and len(val_windows)
            else None
        )

        rng = np.random.default_rng(seed)
        last_loss = float("inf")
        for ep in range(epochs):
            perm = rng.permutation(len(train_flat))
            losses = []
            for i in range(0, len(train_flat), batch_size):
                b = train_flat[perm[i : i + batch_size]]
                if len(b):
                    losses.append(self._model.train_step(b))
            last_loss = float(np.mean(losses)) if losses else float("inf")
            print(f"  epoch {ep + 1:2d}/{epochs}  loss={last_loss:.6f}")

        sample = train_flat[
            np.linspace(0, len(train_flat) - 1, min(2000, len(train_flat)), dtype=int)
        ]
        errors = [self._model.reconstruction_error(w) for w in sample]
        self._model.calibrate_threshold(errors, safety_mult=self._safety_mult)

        val_fpr = None
        if val_flat is not None and len(val_flat):
            val_err = np.array([self._model.reconstruction_error(w) for w in val_flat])
            val_fpr = float((val_err > self._model.threshold).mean())
            print(f"[Calibrate] held-out normal FPR @ threshold = {val_fpr:.4f}")

        return FitResult(
            threshold=float(self._model.threshold),
            train_loss=last_loss,
            val_fpr=val_fpr,
        )

    def score(self, windows: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        _ = timestamps
        return self._model.reconstruction_errors(self._as_flat(windows)).astype(np.float64)

    @property
    def threshold(self) -> float:
        return float(self._model.threshold)

    def save(self, outdir: Path, name: str) -> None:
        self._model.checkpoint_dir = Path(outdir)
        self._model.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_checkpoint(name)

    def num_params(self) -> int:
        return int(sum(np.prod(np.array(v).shape) for v in self._model.params.values()))

    @staticmethod
    def _as_flat(windows: np.ndarray) -> np.ndarray:
        if windows.ndim == 2:
            return windows.astype(np.float32)
        return flatten_windows(windows)
