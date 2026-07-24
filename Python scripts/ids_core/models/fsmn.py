"""FSMN-AE adapter."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from models.fsmn import (
    DROPOUT_RATE,
    ENC_DIMS,
    FSMN_ORDER,
    FSMN_ORDER_FWD,
    FSMNAE,
    FSMNAEConfig,
    L1_LAMBDA,
    THRESHOLD_K,
)

from .base import FitResult, ModelSpec

SPEC = ModelSpec(
    key="fsmn",
    display_name="FSMN-AE",
    default_ckpt_name="fsmn",
    default_epochs=50,
    default_batch_size=256,
)


class FSMNAdapter:
    spec = SPEC

    def __init__(
        self,
        *,
        n_features: int,
        window_size: int,
        seed: int = 42,
        fsmn_order: int = FSMN_ORDER,
        fsmn_order_fwd: int = FSMN_ORDER_FWD,
        dropout: float = DROPOUT_RATE,
        k: float = THRESHOLD_K,
        l1: float = L1_LAMBDA,
        lr: float = 1e-3,
        **_: Any,
    ):
        self._k, self._l1, self._lr = k, l1, lr
        cfg = FSMNAEConfig(
            window=window_size,
            n_features=n_features,
            enc_dims=ENC_DIMS,
            code_dim=ENC_DIMS[-1],
            fsmn_order=fsmn_order,
            fsmn_order_fwd=fsmn_order_fwd,
            dropout=dropout,
            activation="relu",
        )
        self._model = FSMNAE(cfg, seed=seed)

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
        l1 = float(kwargs.get("l1", self._l1))
        k = float(kwargs.get("k", self._k))
        print(f"[Model] params={self._model.num_params():,}")
        t0 = time.perf_counter()
        self._model.train(
            train_windows, epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=0.0, l1_lambda=l1, seed=seed, verbose=True,
        )
        print(f"[Train] done in {time.perf_counter() - t0:.1f}s")
        info = self._model.build_detector(train_windows, k=k)
        print(f"[Calibrate] theta={info['recon_threshold']:.6f}")
        val_fpr = None
        if val_windows is not None and len(val_windows):
            val_err = self._model.reconstruction_error(val_windows)
            val_fpr = float((val_err > info["recon_threshold"]).mean())
            print(f"[Calibrate] held-out normal FPR={val_fpr:.4f}")
        return FitResult(threshold=float(info["recon_threshold"]), val_fpr=val_fpr, extras=info)

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
        return int(self._model.num_params())
