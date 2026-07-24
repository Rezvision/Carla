"""Shared reconstruction-model interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class ModelSpec:
    key: str
    display_name: str
    default_ckpt_name: str
    needs_timestamps: bool = False
    uses_flat_windows: bool = False
    default_epochs: int = 50
    default_batch_size: int = 256


@dataclass
class FitResult:
    threshold: float
    train_loss: Optional[float] = None
    val_fpr: Optional[float] = None
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ReconstructionModel(Protocol):
    spec: ModelSpec

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
    ) -> FitResult: ...

    def score(
        self,
        windows: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray: ...

    @property
    def threshold(self) -> float: ...

    def save(self, outdir: Path, name: str) -> None: ...

    def num_params(self) -> int: ...
