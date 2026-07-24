"""Shared training orchestration — dataset + model agnostic."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .data import load_dataset
from .models import build_model, get_spec
from .models.base import FitResult, ReconstructionModel
from .profiles import DatasetProfile, get_profile


@dataclass
class TrainConfig:
    dataset: str
    model: str
    data: Optional[Path] = None
    outdir: Optional[Path] = None
    name: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    stride: Optional[int] = None
    window: Optional[int] = None
    val_ratio: float = 0.15
    max_windows: Optional[int] = None
    seed: int = 42
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    fit_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    dataset: str
    model_key: str
    name: str
    outdir: Path
    fit: FitResult
    n_train: int
    n_val: int
    n_params: int
    scaler_path: Path
    meta_path: Path
    profile: DatasetProfile


def prepare_splits(
    ds: dict,
    *,
    val_ratio: float,
    max_windows: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows = ds["windows"]
    timestamps = ds["timestamps"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(windows))
    windows, timestamps = windows[perm], timestamps[perm]
    if max_windows and len(windows) > max_windows:
        windows = windows[:max_windows]
        timestamps = timestamps[:max_windows]
        print(f"[Windows] capped to {len(windows):,}")
    cut = int(len(windows) * (1.0 - val_ratio))
    return windows[:cut], windows[cut:], timestamps[:cut], timestamps[cut:]


def _default_name(dataset: str, model: str, spec_default: str) -> str:
    # Preserve existing kaggle_* names for kaggle compare notebooks.
    if dataset == "kaggle":
        return f"{model}_kaggle" if not spec_default.endswith("_kaggle") else spec_default
    if dataset == "carla":
        # Match historical CARLA names where useful.
        return {"gru": "central", "fsmn": "fsmn_ae", "vae": "vae", "tet": "tet_ae"}.get(
            model, f"{model}_carla"
        )
    return f"{model}_{dataset}"


def _default_outdir(dataset: str, model: str, profile: DatasetProfile) -> Path:
    """All research checkpoints land under experiments/checkpoints/{dataset}/.

    The live edge client still uses ``/tmp/fed_ids_checkpoints`` when deployed;
    pass ``--outdir /tmp/fed_ids_checkpoints --name central`` for that path.
    """
    _ = model
    return profile.checkpoint_dir


def train(cfg: TrainConfig) -> TrainResult:
    profile = get_profile(cfg.dataset)
    spec = get_spec(cfg.model)

    data = Path(cfg.data) if cfg.data else profile.default_data_dir
    outdir = Path(cfg.outdir) if cfg.outdir else _default_outdir(cfg.dataset, cfg.model, profile)
    name = cfg.name or _default_name(cfg.dataset, cfg.model, spec.default_ckpt_name)
    epochs = cfg.epochs if cfg.epochs is not None else spec.default_epochs
    batch_size = cfg.batch_size if cfg.batch_size is not None else spec.default_batch_size
    stride = profile.stride if cfg.stride is None else cfg.stride
    window = profile.window_size if cfg.window is None else cfg.window

    if not data.exists():
        raise SystemExit(f"Data not found: {data}")

    print(f"[Train] dataset={profile.display_name} ({profile.key})")
    print(f"[Train] model={spec.display_name} ({spec.key})  name={name}")
    print(f"[Train] features={profile.n_features}  window={window}  stride={stride}")
    print(f"[Train] cols={list(profile.features)}")

    ds = load_dataset(
        data,
        profile,
        normal_only=True,
        stride=stride,
        window=window,
        fit_scaler_on="normal",
    )
    train_w, val_w, train_ts, val_ts = prepare_splits(
        ds, val_ratio=cfg.val_ratio, max_windows=cfg.max_windows, seed=cfg.seed,
    )
    print(f"[Split] train={len(train_w):,}  val={len(val_w):,}")

    model_kwargs = {
        "n_features": profile.n_features,
        "window_size": window,
        "seed": cfg.seed,
        **cfg.model_kwargs,
    }
    model_kwargs.setdefault("safety_mult", profile.anomaly_safety_mult)
    if cfg.model in ("gru", "can_gru"):
        model_kwargs.setdefault("checkpoint_dir", outdir)

    model: ReconstructionModel = build_model(cfg.model, **model_kwargs)
    fit = model.fit(
        train_w,
        val_windows=val_w if len(val_w) else None,
        train_timestamps=train_ts if spec.needs_timestamps else None,
        val_timestamps=val_ts if spec.needs_timestamps else None,
        epochs=epochs,
        batch_size=batch_size,
        seed=cfg.seed,
        **cfg.fit_kwargs,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    model.save(outdir, name)

    scaler_path = outdir / f"{name}_scaler.npz"
    np.savez(
        scaler_path,
        mean=ds["mean"],
        std=ds["std"],
        features=np.array(profile.features),
    )
    meta = {
        "dataset": profile.key,
        "model": cfg.model,
        "display_name": spec.display_name,
        "features": list(profile.features),
        "n_features": profile.n_features,
        "window_size": window,
        "stride": stride,
        "threshold": fit.threshold,
        "safety_mult": model_kwargs.get("safety_mult"),
        "n_params": model.num_params(),
        "n_train": len(train_w),
        "n_val": len(val_w),
        "needs_timestamps": spec.needs_timestamps,
        "scaler": str(scaler_path),
        "outdir": str(outdir),
        "name": name,
    }
    if fit.val_fpr is not None:
        meta["val_fpr"] = fit.val_fpr
    meta_path = outdir / f"{name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[Done] bundle → {outdir.resolve()} ({name}_*)")
    print(f"[Done] scaler: {scaler_path}")
    print(f"[Done] meta:   {meta_path}")

    return TrainResult(
        dataset=profile.key,
        model_key=cfg.model,
        name=name,
        outdir=outdir,
        fit=fit,
        n_train=len(train_w),
        n_val=len(val_w),
        n_params=model.num_params(),
        scaler_path=scaler_path,
        meta_path=meta_path,
        profile=profile,
    )
