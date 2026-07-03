#!/usr/bin/env python3
"""
train_central.py — centralized (non-federated) training of the GRU autoencoder.

Pools all parquet files into one dataset, trains a single model, calibrates the
anomaly threshold, and saves a checkpoint (+ scaler stats). No MQTT, no FedAvg.

Run from Python scripts/mvp_v1/:
    pip install numpy pandas pyarrow paho-mqtt "jax[cpu]"

    # Fast sanity run on one file
    python train_central.py ../../Data/new_data_processed/vehicle_1_20260321_145906.parquet --epochs 10

    # Full pooled dataset, stride 20 (recommended)
    python train_central.py ../../Data/new_data_processed --stride 20 --epochs 20

    # Cap windows if it is still too slow
    python train_central.py ../../Data/new_data_processed --stride 20 --max-windows 100000 --epochs 20

Notes
-----
* The checkpoint stores weights + threshold only. The mean/std used to normalise
  the data is saved separately as <name>_scaler.npz. To deploy a centrally
  trained model faithfully, the edge client must normalise incoming CAN data with
  these same mean/std values (otherwise the calibrated threshold is invalid).
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

# Same model + constants the live client uses (JAX version — JIT-compiled).
try:
    from fed_client_jax import (
        GRUAutoencoder,
        WINDOW_SIZE,
        N_FEATURES,
        BATCH_SIZE,
        ANOMALY_PERCENTILE,
        ANOMALY_SAFETY_MULT,
    )
except ImportError as e:
    sys.exit(f"Cannot import fed_client_jax — run from the same directory.  ({e})")


# ──────────────────────────────────────────────────────────────────────────────
# Parquet loading
# ──────────────────────────────────────────────────────────────────────────────

FEATURES = ("speed_kmh", "battery_level", "throttle", "brake",
            "steering", "gear", "location_x", "location_y")
assert len(FEATURES) == N_FEATURES

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
          ", ".join(f"{k}←{v}" for k, v in chosen.items()))
    dropped = [c for c in df.columns if c not in chosen.values()]
    if dropped:
        print("[Data] dropped:        " + ", ".join(dropped))

    arr = df[[chosen[c] for c in FEATURES]].to_numpy(dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def to_windows(trace: np.ndarray, stride: int):
    """Normalise (global mean/std) and build (N, 160) flat sliding windows."""
    mu = trace.mean(axis=0)
    sd = np.where(trace.std(axis=0) > 1e-6, trace.std(axis=0), 1.0)
    norm = ((trace - mu) / sd).astype(np.float32)
    idxs = range(0, len(norm) - WINDOW_SIZE + 1, stride)
    windows = np.stack([norm[i:i + WINDOW_SIZE].flatten() for i in idxs])
    return windows, mu.astype(np.float32), sd.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Centralized GRU autoencoder training")
    ap.add_argument("data", type=Path, help="parquet file or directory")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--stride", type=int, default=20,
                    help="window stride (20 matches the live client; 1 = fully overlapping)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap number of training windows (uniform subsample across trace)")
    ap.add_argument("--name", default="central",
                    help="checkpoint name (saved as <name>.npz in the model CHECKPOINT_DIR)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Load + window ─────────────────────────────────────────────────────
    trace = load_parquet(args.data)
    windows, mu, sd = to_windows(trace, args.stride)
    print(f"[Windows] {len(windows):,} windows (stride={args.stride})")

    if args.max_windows and len(windows) > args.max_windows:
        idx = np.linspace(0, len(windows) - 1, args.max_windows, dtype=int)
        windows = windows[idx]
        print(f"[Windows] subsampled to {len(windows):,} (uniform across trace)")

    # ── Train ─────────────────────────────────────────────────────────────
    model = GRUAutoencoder(seed=args.seed)
    print(f"[Train] {args.epochs} epochs, batch={args.batch_size}")
    for ep in range(args.epochs):
        perm = rng.permutation(len(windows))
        losses = []
        t0 = time.perf_counter()
        for i in range(0, len(windows), args.batch_size):
            b = windows[perm[i:i + args.batch_size]]
            if len(b):
                losses.append(model.train_step(b))
        avg = float(np.mean(losses)) if losses else float("inf")
        print(f"  epoch {ep+1:2d}/{args.epochs}  "
              f"loss={avg:.6f}  ({time.perf_counter() - t0:.1f}s)")

    # ── Calibrate threshold on a uniform sample of normal windows ─────────
    sample = windows[np.linspace(0, len(windows) - 1,
                                 min(2000, len(windows)), dtype=int)]
    errors = [model.reconstruction_error(w) for w in sample]
    model.calibrate_threshold(errors)
    print(f"[Calibrate] p{ANOMALY_PERCENTILE} × {ANOMALY_SAFETY_MULT} "
          f"→ threshold={model.threshold:.6f}")

    # ── Save checkpoint + scaler ─────────────────────────────────────────
    ckpt = model.save_checkpoint(args.name)
    scaler_path = Path(ckpt).with_name(f"{args.name}_scaler.npz")
    np.savez(scaler_path, mean=mu, std=sd)
    print(f"[Done] checkpoint: {ckpt}")
    print(f"[Done] scaler:     {scaler_path}")


if __name__ == "__main__":
    main()
