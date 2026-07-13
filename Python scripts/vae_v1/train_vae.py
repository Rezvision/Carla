#!/usr/bin/env python3
"""
train_vae.py — centralized training of the paper's LSTM-VAE on translated CAN data.

Pools parquet telemetry, z-score standardises it, trains the β-VAE, then builds and
calibrates the three anomaly-detection heads (reconstruction, latent distance,
latent clustering) on a held-out normal validation split. Saves a self-contained
model bundle (weights + scaler + thresholds + BallTree data + Isolation Forest).

Run from Python scripts/vae_v1/:

    pip install "jax[cpu]" flax optax scikit-learn pandas pyarrow joblib

    # Fast sanity run on one file
    python train_vae.py ../../Data/new_data_processed/vehicle_1_20260321_145906.parquet \
        --epochs 10

    # Full pooled dataset, reconstruction objective (beta=0.8, paper default)
    python train_vae.py ../../Data/new_data_processed --epochs 100

    # Latent-space objective (beta=2.0) for the encoder-only detectors
    python train_vae.py ../../Data/new_data_processed --objective latent --epochs 100

Notes
-----
* Sequence windows are shape (N, WINDOW_SIZE, N_FEATURES) — the VAE consumes the
  raw sequence, unlike the flat GRU autoencoder in mvp_v1.
* The train/val split is sequential (no shuffling across the boundary) so the
  validation set is genuinely unseen normal traffic, matching the paper's
  threshold-calibration protocol.
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

from vae_model import (
    LSTMVAE, VAEConfig, FEATURES, N_FEATURES, WINDOW_SIZE, STRIDE,
    BETA_RECON, BETA_LATENT, N_SAMPLES_TRAIN, THRESHOLD_MARGIN,
    IF_CONTAMINATION,
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
    ap = argparse.ArgumentParser(description="Centralized LSTM-VAE training (paper model)")
    ap.add_argument("data", type=Path, help="parquet file or directory")
    ap.add_argument("--objective", choices=("recon", "latent"), default="recon",
                    help="recon -> beta=0.8 (reconstruction head), "
                         "latent -> beta=2.0 (latent-space heads)")
    ap.add_argument("--beta", type=float, default=None,
                    help="override beta explicitly")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--lstm-units", type=int, default=64)
    ap.add_argument("--activation", choices=("tanh", "relu"), default="tanh",
                    help="LSTM activation. Paper specifies relu; tanh (default) is "
                         "numerically stable over length-50 sequences.")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES_TRAIN)
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="fraction of the (sequential) tail used for threshold calibration")
    ap.add_argument("--margin", type=float, default=THRESHOLD_MARGIN,
                    help="threshold margin factor gamma (tau = max * (1+gamma))")
    ap.add_argument("--contamination", type=float, default=IF_CONTAMINATION)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap training windows (uniform subsample)")
    ap.add_argument("--outdir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--name", default="vae")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    beta = args.beta if args.beta is not None else (
        BETA_RECON if args.objective == "recon" else BETA_LATENT)
    print(f"[Config] objective={args.objective}  beta={beta}  "
          f"window={WINDOW_SIZE}  stride={args.stride}  latent=10  "
          f"lstm_units={args.lstm_units}")

    rng = np.random.default_rng(args.seed)

    # ── Load, standardise, window (fit scaler on TRAIN portion only) ───────────
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

    if len(train_w) == 0 or len(val_w) == 0:
        sys.exit("Not enough rows to build both train and validation windows. "
                 "Use more data or a smaller --stride / --val-ratio.")

    if args.max_windows and len(train_w) > args.max_windows:
        idx = np.linspace(0, len(train_w) - 1, args.max_windows, dtype=int)
        train_w = train_w[idx]
        print(f"[Windows] subsampled train to {len(train_w):,}")

    # ── Train ──────────────────────────────────────────────────────────────────
    cfg = VAEConfig(window=WINDOW_SIZE, n_features=N_FEATURES,
                    lstm_units=args.lstm_units, latent_dim=10,
                    activation=args.activation)
    model = LSTMVAE(cfg, seed=args.seed)
    print(f"[Train] {args.epochs} epochs, batch={args.batch_size}, "
          f"multi-sampling={args.n_samples}")
    t0 = time.perf_counter()
    model.train(train_w, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay, beta=beta,
                n_samples=args.n_samples, seed=args.seed, verbose=True)
    print(f"[Train] done in {time.perf_counter() - t0:.1f}s")

    # ── Build + calibrate the three detection heads ────────────────────────────
    info = model.build_detectors(train_w, val_w, margin=args.margin,
                                 contamination=args.contamination)
    print(f"[Calibrate] reconstruction threshold = {info['recon_threshold']:.6f}")
    print(f"[Calibrate] latent-distance threshold = {info['dist_threshold']:.6f}")
    print(f"[Calibrate] latent means stored       = {info['n_train_means']:,}")

    # ── Save bundle ────────────────────────────────────────────────────────────
    outdir = args.outdir
    model.save(str(outdir), name=args.name)
    np.savez(outdir / f"{args.name}_scaler.npz", mean=mu.astype(np.float32),
             std=sd.astype(np.float32))
    print(f"[Done] model bundle saved to: {outdir.resolve()}")
    print(f"       {args.name}_params.msgpack, {args.name}_meta.json, "
          f"{args.name}_scaler.npz, {args.name}_train_means.npy, "
          f"{args.name}_iforest.joblib")


if __name__ == "__main__":
    main()
