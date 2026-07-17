#!/usr/bin/env python3
"""
train_tet.py — centralized training of AE+TET on translated CAN data.

Pools parquet telemetry, z-score standardises it, trains the AE + Temporal Embedding
Transformer reconstruction model (Tao & Xiyang 2025, without federated learning), then
calibrates theta = mu + k*sigma on normal training reconstruction errors.

Run from Python scripts/tet_v1/:

    pip install "jax[cpu]" flax optax scikit-learn pandas pyarrow

    python train_tet.py ../../Data/new_data_processed/vehicle_1_20260321_145906.parquet \
        --epochs 10

    python train_tet.py ../../Data/new_data_processed --epochs 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("Install deps:  pip install pandas pyarrow")

from tet_model import (
    TETAE, TETConfig, FEATURES, N_FEATURES, WINDOW_SIZE, STRIDE,
    AE_LATENT, D_MODEL, N_HEADS, N_LAYERS, FF_DIM, DROPOUT_RATE,
    NOISE_STD, THRESHOLD_K,
)

ALIASES = {
    "speed_kmh":     ("speed_kmh", "speed", "velocity"),
    "battery_level": ("battery_level", "battery", "soc"),
    "throttle":      ("throttle", "throttle_pct"),
    "brake":         ("brake", "brake_pct"),
    "steering":      ("steering", "steer", "steering_angle"),
    "gear":          ("gear", "current_gear"),
    "location_x":    ("location_x", "loc_x", "pos_x", "x"),
    "location_y":    ("location_y", "loc_y", "pos_y", "y"),
}

TS_ALIASES = ("timestamp", "time", "ts", "t")


def load_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [N,F], normalised timestamps [N] in [0, 1])."""
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not files:
        sys.exit(f"No parquet files at {path}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"[Data] loaded {len(df):,} rows from {len(files)} file(s)")

    cols_lower = {c.lower(): c for c in df.columns}
    chosen: dict[str, str] = {}
    for canonical in FEATURES:
        for alias in ALIASES[canonical]:
            if alias.lower() in cols_lower:
                chosen[canonical] = cols_lower[alias.lower()]
                break
        else:
            sys.exit(f"Missing required feature {canonical!r}.  "
                     f"Available: {list(df.columns)}")

    print("[Data] using columns:  " +
          ", ".join(f"{k}<-{v}" for k, v in chosen.items()))
    arr = df[[chosen[c] for c in FEATURES]].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    ts_col = None
    for alias in TS_ALIASES:
        if alias in cols_lower:
            ts_col = cols_lower[alias]
            break
    if ts_col is not None:
        series = df[ts_col]
        if pd.api.types.is_numeric_dtype(series):
            raw_ts = series.to_numpy(dtype=np.float64)
        else:
            parsed = pd.to_datetime(series, errors="coerce")
            # nanoseconds since epoch; NaT → NaN
            raw_ts = parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(
                np.float64)
            raw_ts[pd.isna(parsed).to_numpy()] = np.nan
        tmin = float(np.nanmin(raw_ts))
        tmax = float(np.nanmax(raw_ts))
        denom = (tmax - tmin) if tmax > tmin else 1.0
        ts = ((raw_ts - tmin) / denom).astype(np.float32)          # paper Eq. 7
        print(f"[Data] timestamps from column {ts_col!r} (normalised to [0,1])")
    else:
        n = len(arr)
        ts = (np.arange(n, dtype=np.float32) / max(n - 1, 1))
        print("[Data] no timestamp column — using normalised row index for TE")

    return arr, np.nan_to_num(ts, nan=0.0, posinf=1.0, neginf=0.0)


def make_windows(norm: np.ndarray, ts: np.ndarray, stride: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, WINDOW, F) feature windows and (N, WINDOW) timestamp windows."""
    idxs = list(range(0, len(norm) - WINDOW_SIZE + 1, stride))
    if not idxs:
        return (np.zeros((0, WINDOW_SIZE, norm.shape[1]), dtype=np.float32),
                np.zeros((0, WINDOW_SIZE), dtype=np.float32))
    feat_w = np.stack([norm[i:i + WINDOW_SIZE] for i in idxs]).astype(np.float32)
    ts_w = np.stack([ts[i:i + WINDOW_SIZE] for i in idxs]).astype(np.float32)
    return feat_w, ts_w


def main():
    ap = argparse.ArgumentParser(description="Centralized AE+TET training")
    ap.add_argument("data", type=Path, help="parquet file or directory")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--d-model", type=int, default=D_MODEL)
    ap.add_argument("--n-heads", type=int, default=N_HEADS)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--ff-dim", type=int, default=FF_DIM)
    ap.add_argument("--ae-latent", type=int, default=AE_LATENT)
    ap.add_argument("--dropout", type=float, default=DROPOUT_RATE)
    ap.add_argument("--noise-std", type=float, default=NOISE_STD,
                    help="Gaussian input noise for denoising AE (paper: 0.1)")
    ap.add_argument("--k", type=float, default=THRESHOLD_K)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--name", default="tet_ae")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[Config] window={WINDOW_SIZE}  stride={args.stride}  "
          f"ae_latent={args.ae_latent}  d_model={args.d_model}  "
          f"heads={args.n_heads}  layers={args.n_layers}  "
          f"noise={args.noise_std}  k={args.k}")

    trace, ts = load_parquet(args.data)
    cut = int(len(trace) * (1.0 - args.val_ratio))
    train_trace, val_trace = trace[:cut], trace[cut:]
    train_ts, val_ts = ts[:cut], ts[cut:]

    mu = train_trace.mean(axis=0)
    sd = np.where(train_trace.std(axis=0) > 1e-6, train_trace.std(axis=0), 1.0)
    train_norm = ((train_trace - mu) / sd).astype(np.float32)
    val_norm = ((val_trace - mu) / sd).astype(np.float32)

    train_w, train_w_ts = make_windows(train_norm, train_ts, args.stride)
    val_w, val_w_ts = make_windows(val_norm, val_ts, args.stride)
    print(f"[Windows] train={len(train_w):,}  val={len(val_w):,}")

    if len(train_w) == 0:
        sys.exit("Not enough rows to build training windows.")

    if args.max_windows and len(train_w) > args.max_windows:
        idx = np.linspace(0, len(train_w) - 1, args.max_windows, dtype=int)
        train_w, train_w_ts = train_w[idx], train_w_ts[idx]
        print(f"[Windows] subsampled train to {len(train_w):,}")

    cfg = TETConfig(
        window=WINDOW_SIZE, n_features=N_FEATURES,
        ae_latent=args.ae_latent, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, ff_dim=args.ff_dim,
        dropout=args.dropout, noise_std=args.noise_std,
    )
    model = TETAE(cfg, seed=args.seed)
    print(f"[Model] parameters = {model.num_params():,}  "
          f"(~{model.approx_size_kb():.1f} KiB float32)")
    print(f"[Train] {args.epochs} epochs, batch={args.batch_size}")
    t0 = time.perf_counter()
    model.train(train_w, train_w_ts, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay, noise_std=args.noise_std,
                seed=args.seed, verbose=True)
    print(f"[Train] done in {time.perf_counter() - t0:.1f}s")

    info = model.build_detector(train_w, train_w_ts, k=args.k)
    print(f"[Calibrate] mu={info['err_mean']:.6f}  sigma={info['err_std']:.6f}  "
          f"k={info['k']}  ->  theta = {info['recon_threshold']:.6f}")
    if len(val_w):
        val_err = model.reconstruction_error(val_w, val_w_ts)
        val_fpr = float((val_err > info["recon_threshold"]).mean())
        print(f"[Calibrate] held-out normal FPR at theta = {val_fpr:.4f}")

    outdir = args.outdir
    model.save(str(outdir), name=args.name)
    np.savez(outdir / f"{args.name}_scaler.npz", mean=mu.astype(np.float32),
             std=sd.astype(np.float32))
    print(f"[Done] model bundle saved to: {outdir.resolve()}")
    print(f"       {args.name}_params.msgpack, {args.name}_meta.json, "
          f"{args.name}_scaler.npz")


if __name__ == "__main__":
    main()
