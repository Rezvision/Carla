#!/usr/bin/env python3
"""
train_fsmn.py — centralized training of the FSMN-AE on translated CAN data.

Pools parquet telemetry, z-score standardises it, trains the FSMN autoencoder, then
calibrates the paper's dynamic reconstruction threshold theta = mu + k*sigma on the
*normal* training reconstruction-error distribution (Zhou et al. 2026, Eq. 4). Saves a
self-contained bundle (weights + scaler + threshold + config).

Run from Python scripts/fsmn_ae_v1/:

    pip install "jax[cpu]" flax optax scikit-learn pandas pyarrow

    # Fast sanity run on one file
    python train_fsmn.py ../../Data/new_data_processed/vehicle_1_20260321_145906.parquet \
        --epochs 10

    # Full pooled dataset (paper defaults: N=10 memory, lambda=0.001, k=2.5)
    python train_fsmn.py ../../Data/new_data_processed --epochs 100

Notes
-----
* Sequence windows are shape (N, WINDOW_SIZE, N_FEATURES) — identical windowing to the
  sibling vae_v1 model so the two are directly comparable.
* The scaler is fitted on the training portion only; the tail split is held out purely
  to report an honest normal-traffic error distribution (the threshold itself follows the
  paper and is calibrated on the training errors).
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

from fsmn_model import (
    FSMNAE, FSMNAEConfig, FEATURES, N_FEATURES, WINDOW_SIZE, STRIDE,
    ENC_DIMS, FSMN_ORDER, FSMN_ORDER_FWD, L1_LAMBDA, DROPOUT_RATE, THRESHOLD_K,
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


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def load_parquet(path: Path) -> np.ndarray:
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
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def make_windows(norm: np.ndarray, stride: int) -> np.ndarray:
    """Build (N, WINDOW_SIZE, F) sequence windows from a normalised trace."""
    idxs = range(0, len(norm) - WINDOW_SIZE + 1, stride)
    if not idxs:
        return np.zeros((0, WINDOW_SIZE, norm.shape[1]), dtype=np.float32)
    return np.stack([norm[i:i + WINDOW_SIZE] for i in idxs]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Centralized FSMN-AE training (paper model)")
    ap.add_argument("data", type=Path, help="parquet file or directory")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--fsmn-order", type=int, default=FSMN_ORDER,
                    help="N backward memory taps (paper: 10)")
    ap.add_argument("--fsmn-order-fwd", type=int, default=FSMN_ORDER_FWD,
                    help="forward look-ahead taps (>0 => bidirectional memory)")
    ap.add_argument("--l1", type=float, default=L1_LAMBDA,
                    help="L1 sparsity penalty on the latent code (paper: 0.001)")
    ap.add_argument("--dropout", type=float, default=DROPOUT_RATE)
    ap.add_argument("--activation", choices=("relu", "tanh"), default="relu",
                    help="encoder/fusion activation (paper: relu)")
    ap.add_argument("--k", type=float, default=THRESHOLD_K,
                    help="dynamic-threshold factor: theta = mu + k*sigma (paper: 2.5)")
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="fraction of the (sequential) tail held out as unseen normal data")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap training windows (uniform subsample)")
    ap.add_argument("--outdir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--name", default="fsmn_ae")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[Config] window={WINDOW_SIZE}  stride={args.stride}  enc_dims={ENC_DIMS}  "
          f"fsmn_order={args.fsmn_order}(+{args.fsmn_order_fwd})  l1={args.l1}  "
          f"dropout={args.dropout}  act={args.activation}  k={args.k}")

    # ── Load, standardise, window (scaler fitted on TRAIN portion only) ─────────
    trace = load_parquet(args.data)
    cut = int(len(trace) * (1.0 - args.val_ratio))
    train_trace, val_trace = trace[:cut], trace[cut:]

    mu = train_trace.mean(axis=0)
    sd = np.where(train_trace.std(axis=0) > 1e-6, train_trace.std(axis=0), 1.0)

    train_norm = ((train_trace - mu) / sd).astype(np.float32)
    val_norm = ((val_trace - mu) / sd).astype(np.float32)

    train_w = make_windows(train_norm, args.stride)
    val_w = make_windows(val_norm, args.stride)
    print(f"[Windows] train={len(train_w):,}  val={len(val_w):,}  "
          f"(z-scored, scaler fitted on train)")

    if len(train_w) == 0:
        sys.exit("Not enough rows to build training windows. "
                 "Use more data or a smaller --stride.")

    if args.max_windows and len(train_w) > args.max_windows:
        idx = np.linspace(0, len(train_w) - 1, args.max_windows, dtype=int)
        train_w = train_w[idx]
        print(f"[Windows] subsampled train to {len(train_w):,}")

    # ── Train ──────────────────────────────────────────────────────────────────
    cfg = FSMNAEConfig(window=WINDOW_SIZE, n_features=N_FEATURES, enc_dims=ENC_DIMS,
                       code_dim=ENC_DIMS[-1], fsmn_order=args.fsmn_order,
                       fsmn_order_fwd=args.fsmn_order_fwd, dropout=args.dropout,
                       activation=args.activation)
    model = FSMNAE(cfg, seed=args.seed)
    print(f"[Model] parameters = {model.num_params():,}  "
          f"(~{model.approx_size_kb():.1f} KiB float32)")
    print(f"[Train] {args.epochs} epochs, batch={args.batch_size}")
    t0 = time.perf_counter()
    model.train(train_w, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, l1_lambda=args.l1, seed=args.seed,
                verbose=True)
    print(f"[Train] done in {time.perf_counter() - t0:.1f}s")

    # ── Calibrate dynamic threshold (paper Eq. 4) ──────────────────────────────
    info = model.build_detector(train_w, k=args.k)
    print(f"[Calibrate] mu={info['err_mean']:.6f}  sigma={info['err_std']:.6f}  "
          f"k={info['k']}  ->  theta = {info['recon_threshold']:.6f}")
    if len(val_w):
        val_err = model.reconstruction_error(val_w)
        val_fpr = float((val_err > info["recon_threshold"]).mean())
        print(f"[Calibrate] held-out normal FPR at theta = {val_fpr:.4f}")

    # ── Save bundle ────────────────────────────────────────────────────────────
    outdir = args.outdir
    model.save(str(outdir), name=args.name)
    np.savez(outdir / f"{args.name}_scaler.npz", mean=mu.astype(np.float32),
             std=sd.astype(np.float32))
    print(f"[Done] model bundle saved to: {outdir.resolve()}")
    print(f"       {args.name}_params.msgpack, {args.name}_meta.json, "
          f"{args.name}_scaler.npz")


if __name__ == "__main__":
    main()
