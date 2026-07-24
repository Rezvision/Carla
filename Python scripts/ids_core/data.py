"""Unified parquet loading / windowing for any DatasetProfile."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("Install deps: pip install pandas pyarrow")

from .profiles import DatasetProfile


def list_parquet(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if files:
            return files
    raise SystemExit(f"No parquet files at {path}")


def resolve_feature_columns(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    chosen: list[str] = []
    for canonical in profile.features:
        aliases = profile.aliases.get(canonical, (canonical,))
        for alias in aliases:
            if alias.lower() in cols_lower:
                chosen.append(cols_lower[alias.lower()])
                break
        else:
            raise SystemExit(
                f"Missing feature {canonical!r} for dataset={profile.key}. "
                f"Tried {aliases}. Available: {list(df.columns)}"
            )
    return chosen


def load_traces(path: Path, profile: DatasetProfile) -> list[pd.DataFrame]:
    """Return one dataframe per independent time series."""
    frames: list[pd.DataFrame] = []
    for f in list_parquet(path):
        df = pd.read_parquet(f)
        feat_cols = resolve_feature_columns(df, profile)

        # Keep useful metadata when present.
        keep = list(dict.fromkeys(
            feat_cols
            + [c for c in (
                profile.vehicle_id_col,
                profile.label_col,
                profile.failure_type_col,
                *[a for a in profile.timestamp_aliases if a in df.columns],
            ) if c and c in df.columns]
        ))
        # Also keep first matching timestamp under a normalised name.
        ts_col = next((cols for cols in profile.timestamp_aliases if cols in df.columns), None)
        if ts_col and "timestamp" not in keep:
            # rename path handled below
            pass

        sub = df[keep].copy()
        if ts_col and ts_col != "timestamp":
            sub = sub.rename(columns={ts_col: "timestamp"})
        elif ts_col == "timestamp" and "timestamp" not in sub.columns and ts_col in df.columns:
            sub["timestamp"] = df[ts_col]

        if "timestamp" in sub.columns:
            sub = sub.sort_values("timestamp")

        vid = profile.vehicle_id_col
        if vid and vid in sub.columns and sub[vid].nunique() > 1:
            for _, g in sub.groupby(vid, sort=True):
                frames.append(g.reset_index(drop=True))
        else:
            frames.append(sub.reset_index(drop=True))
    return frames


def feature_matrix(df: pd.DataFrame, profile: DatasetProfile) -> np.ndarray:
    cols = resolve_feature_columns(df, profile)
    arr = df[cols].to_numpy(dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def anomaly_mask(df: pd.DataFrame, profile: DatasetProfile) -> np.ndarray:
    if not profile.has_labels:
        return np.zeros(len(df), dtype=np.int8)
    if profile.label_col and profile.label_col in df.columns:
        return df[profile.label_col].to_numpy(dtype=np.int8)
    if profile.failure_type_col and profile.failure_type_col in df.columns:
        return (df[profile.failure_type_col].astype(str) != "No Failure").to_numpy(dtype=np.int8)
    return np.zeros(len(df), dtype=np.int8)


def timestamp_vector(df: pd.DataFrame) -> np.ndarray:
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
    *,
    stride: int,
    window: int,
    labels: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(trace) < window:
        return (
            np.zeros((0, window, trace.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0, window), dtype=np.float32),
        )
    idxs = list(range(0, len(trace) - window + 1, stride))
    windows = np.stack([trace[i : i + window] for i in idxs]).astype(np.float32)
    if labels is None:
        y = np.zeros(len(windows), dtype=np.int8)
    else:
        y = np.array([1 if labels[i : i + window].any() else 0 for i in idxs], dtype=np.int8)
    if timestamps is None:
        ts_w = np.zeros((len(windows), window), dtype=np.float32)
    else:
        ts_w = np.stack([timestamps[i : i + window] for i in idxs]).astype(np.float32)
    return windows, y, ts_w


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    n, t, f = windows.shape
    return windows.reshape(n, t * f).astype(np.float32)


def load_dataset(
    path: Path,
    profile: DatasetProfile,
    *,
    normal_only: bool = False,
    stride: Optional[int] = None,
    window: Optional[int] = None,
    fit_scaler_on: str = "normal",
) -> dict:
    """
    Load traces for ``profile``, z-score, and build windows.

    Returns dict with windows, labels, timestamps, mean, std, profile metadata.
    """
    stride = profile.stride if stride is None else stride
    window = profile.window_size if window is None else window

    frames = load_traces(path, profile)
    print(f"[Data] dataset={profile.key}  traces={len(frames)}  from {path}")

    scaler_rows = []
    for df in frames:
        feats = feature_matrix(df, profile)
        mask = anomaly_mask(df, profile)
        if fit_scaler_on == "normal" and profile.has_labels:
            scaler_rows.append(feats[mask == 0] if (mask == 0).any() else feats)
        else:
            scaler_rows.append(feats)
    stacked = (
        np.concatenate(scaler_rows, axis=0)
        if scaler_rows
        else np.zeros((1, profile.n_features), dtype=np.float32)
    )
    mu = stacked.mean(axis=0).astype(np.float32)
    sd = stacked.std(axis=0).astype(np.float32)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)

    all_w, all_y, all_t = [], [], []
    n_rows = 0
    for df in frames:
        feats = feature_matrix(df, profile)
        labels = anomaly_mask(df, profile)
        ts = timestamp_vector(df)
        if normal_only and profile.has_labels:
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
    print(
        f"[Data] rows={n_rows:,}  windows={len(windows):,}  "
        f"anomalous_windows={int(labels.sum())}  "
        f"features={profile.n_features}  window={window}  stride={stride}"
    )
    return {
        "windows": windows,
        "labels": labels,
        "timestamps": timestamps,
        "mean": mu,
        "std": sd,
        "n_features": profile.n_features,
        "window": window,
        "features": profile.features,
        "profile": profile,
    }
