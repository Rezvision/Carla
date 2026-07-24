"""
Shared data loading / windowing for the Kaggle experiment.

Builds windows *per vehicle* (no cross-vehicle stitching), which matters because
each vehicle is a short independent series.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("Install deps: pip install pandas pyarrow")

from datasets.kaggle.config import FEATURES, N_FEATURES, STRIDE, WINDOW_SIZE


def list_parquet(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if files:
            return files
    raise SystemExit(f"No parquet files at {path}")


def load_vehicle_frames(path: Path) -> list[pd.DataFrame]:
    """Return one sorted dataframe per vehicle parquet (or per vehicle_id)."""
    frames: list[pd.DataFrame] = []
    for f in list_parquet(path):
        df = pd.read_parquet(f)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise SystemExit(f"{f.name} missing features: {missing}")
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        if "vehicle_id" in df.columns and df["vehicle_id"].nunique() > 1:
            for _, g in df.groupby("vehicle_id", sort=True):
                frames.append(g.reset_index(drop=True))
        else:
            frames.append(df.reset_index(drop=True))
    return frames


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    arr = df[list(FEATURES)].to_numpy(dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def anomaly_mask(df: pd.DataFrame) -> np.ndarray:
    if "is_anomaly" in df.columns:
        return df["is_anomaly"].to_numpy(dtype=np.int8)
    if "failure_type" in df.columns:
        return (df["failure_type"].astype(str) != "No Failure").to_numpy(dtype=np.int8)
    return np.zeros(len(df), dtype=np.int8)


def timestamp_vector(df: pd.DataFrame) -> np.ndarray:
    """
    Per-vehicle timestamps normalised to [0, 1] for TET temporal embedding.
    Falls back to row index when no usable timestamp column exists.
    """
    n = len(df)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if "timestamp" in df.columns:
        series = df["timestamp"]
        if pd.api.types.is_numeric_dtype(series):
            raw = series.to_numpy(dtype=np.float64)
        else:
            parsed = pd.to_datetime(series, errors="coerce")
            raw = parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.float64)
            raw[pd.isna(parsed).to_numpy()] = np.nan
        tmin = float(np.nanmin(raw)) if np.isfinite(np.nanmin(raw)) else 0.0
        tmax = float(np.nanmax(raw)) if np.isfinite(np.nanmax(raw)) else 1.0
        denom = (tmax - tmin) if tmax > tmin else 1.0
        ts = ((raw - tmin) / denom).astype(np.float32)
        return np.nan_to_num(ts, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.arange(n, dtype=np.float32) / max(n - 1, 1))


def make_windows(
    trace: np.ndarray,
    stride: int = STRIDE,
    window: int = WINDOW_SIZE,
    labels: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding windows over one vehicle trace.

    Returns
    -------
    windows : (N, window, F)
    window_labels : (N,)  1 if any frame in the window is anomalous
    window_ts : (N, window) normalised timestamps (zeros if none provided)
    """
    if len(trace) < window:
        empty_w = np.zeros((0, window, trace.shape[1]), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int8)
        empty_t = np.zeros((0, window), dtype=np.float32)
        return empty_w, empty_y, empty_t

    idxs = list(range(0, len(trace) - window + 1, stride))
    windows = np.stack([trace[i : i + window] for i in idxs]).astype(np.float32)
    if labels is None:
        y = np.zeros(len(windows), dtype=np.int8)
    else:
        y = np.array(
            [1 if labels[i : i + window].any() else 0 for i in idxs],
            dtype=np.int8,
        )
    if timestamps is None:
        ts_w = np.zeros((len(windows), window), dtype=np.float32)
    else:
        ts_w = np.stack([timestamps[i : i + window] for i in idxs]).astype(np.float32)
    return windows, y, ts_w


def load_dataset(
    path: Path,
    *,
    normal_only: bool = False,
    stride: int = STRIDE,
    window: int = WINDOW_SIZE,
    fit_scaler_on: str = "normal",
) -> dict:
    """
    Load all vehicles, z-score, window.

    fit_scaler_on:
      - "normal": mean/std from normal rows only (recommended)
      - "all": mean/std from all loaded rows

    Also returns ``timestamps`` shaped (N, window) for TET temporal embedding.
    """
    frames = load_vehicle_frames(path)
    print(f"[Data] {len(frames)} vehicle trace(s) from {path}")

    # Collect rows for scaler
    scaler_rows = []
    for df in frames:
        feats = feature_matrix(df)
        mask = anomaly_mask(df)
        if fit_scaler_on == "normal":
            scaler_rows.append(feats[mask == 0] if (mask == 0).any() else feats)
        else:
            scaler_rows.append(feats)
    stacked = np.concatenate(scaler_rows, axis=0) if scaler_rows else np.zeros((1, N_FEATURES))
    mu = stacked.mean(axis=0).astype(np.float32)
    sd = stacked.std(axis=0).astype(np.float32)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)

    all_w, all_y, all_t = [], [], []
    n_rows = 0
    for df in frames:
        feats = feature_matrix(df)
        labels = anomaly_mask(df)
        ts = timestamp_vector(df)
        if normal_only:
            keep = labels == 0
            feats, labels, ts = feats[keep], labels[keep], ts[keep]
        n_rows += len(feats)
        norm = ((feats - mu) / sd).astype(np.float32)
        w, y, t = make_windows(
            norm, stride=stride, window=window, labels=labels, timestamps=ts,
        )
        if len(w):
            all_w.append(w)
            all_y.append(y)
            all_t.append(t)

    if not all_w:
        raise SystemExit(
            "No windows produced. Need longer traces, smaller --window, or more data."
        )

    windows = np.concatenate(all_w, axis=0)
    labels = np.concatenate(all_y, axis=0)
    timestamps = np.concatenate(all_t, axis=0)
    print(f"[Data] rows={n_rows:,}  windows={len(windows):,}  "
          f"anomalous_windows={int(labels.sum())}  "
          f"features={N_FEATURES}  window={window}  stride={stride}")
    return {
        "windows": windows,
        "labels": labels,
        "timestamps": timestamps,
        "mean": mu,
        "std": sd,
        "n_features": N_FEATURES,
        "window": window,
        "features": FEATURES,
    }


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """(N, T, F) → (N, T*F) for the flat GRU AE."""
    n, t, f = windows.shape
    return windows.reshape(n, t * f).astype(np.float32)
