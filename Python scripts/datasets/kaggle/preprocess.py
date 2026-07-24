#!/usr/bin/env python3
"""
CSV → per-vehicle parquet for the Kaggle anomaly-detection experiment.

Writes Data/kaggle/processed/<vehicle_id>.parquet with the native FEATURES
plus label/metadata columns. Does not touch CARLA Data/carla/processed.

    cd "Python scripts"
    python -m datasets.kaggle.preprocess
    python -m datasets.kaggle.preprocess --csv ../Data/kaggle/raw/kaggle_synthetic_telemetry_data.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

try:
    import pandas as pd
except ImportError:
    sys.exit("Install deps: pip install pandas pyarrow")

from datasets.kaggle.config import (
    DEFAULT_CSV,
    FALLBACK_CSV,
    FEATURES,
    META_COLS,
    PROCESSED_DIR,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None,
                    help="Source CSV (default: augmented if present, else original)")
    ap.add_argument("--outdir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--normal-only", action="store_true",
                    help="Drop anomalous rows before writing (training-only export)")
    ap.add_argument("--clear", action="store_true",
                    help="Delete existing parquet files in outdir first")
    return ap.parse_args()


def resolve_csv(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_CSV.is_file():
        return DEFAULT_CSV
    if FALLBACK_CSV.is_file():
        return FALLBACK_CSV
    raise SystemExit(f"No CSV found at {DEFAULT_CSV} or {FALLBACK_CSV}")


def main() -> int:
    args = parse_args()
    csv_path = resolve_csv(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing required feature columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "failure_type" not in df.columns:
        df["failure_type"] = "No Failure"
    df["is_anomaly"] = (df["failure_type"].astype(str) != "No Failure").astype("int8")
    for col in ("engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent"):
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int8")

    if args.normal_only:
        before = len(df)
        df = df[df["is_anomaly"] == 0].copy()
        print(f"[Preprocess] normal-only: kept {len(df):,}/{before:,} rows")

    keep = [c for c in ("vehicle_id", "timestamp", *FEATURES,
                        "failure_type", "is_anomaly",
                        "engine_failure_imminent", "brake_issue_imminent",
                        "battery_issue_imminent") if c in df.columns]
    # META_COLS unused beyond documentation; keep list explicit above.
    _ = META_COLS
    df = df[keep].sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for old in outdir.glob("*.parquet"):
            old.unlink()

    n_files = 0
    for vid, g in df.groupby("vehicle_id", sort=True):
        path = outdir / f"{vid}.parquet"
        g.reset_index(drop=True).to_parquet(path, index=False)
        n_files += 1

    n_anom = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    print(f"[Preprocess] source: {csv_path}")
    print(f"[Preprocess] features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"[Preprocess] wrote {n_files} parquet files, {len(df):,} rows "
          f"({n_anom} anomalous) → {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
