#!/usr/bin/env python3
"""
Raw HCRL CAN CSV → engineered parquet for ``can_gru``.

Applies Chowdhury-style feature engineering (ID bits, payload, temporal):

    cd "Python scripts"
    python -m datasets.can.preprocess
    python -m datasets.can.preprocess --include-attacks
    python -m datasets.can.preprocess --csv ../Data/CAN/DoS_dataset.csv --name dos
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

from datasets.can.config import ATTACK_CSVS, NORMAL_CSV, PROCESSED_DIR
from models.can_gru.features import FEATURES, engineer_can_frames, read_raw_can_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Single CSV to convert (default: normal_run_data.csv)",
    )
    ap.add_argument("--name", default=None, help="Output stem (default: from filename)")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Parquet directory (default: {PROCESSED_DIR})",
    )
    ap.add_argument(
        "--include-attacks",
        action="store_true",
        help="Also convert DoS / Fuzzy / RPM / gear attack CSVs",
    )
    ap.add_argument(
        "--normal-only",
        action="store_true",
        help="Drop anomalous rows before writing",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap (debug / smoke tests)",
    )
    ap.add_argument("--clear", action="store_true", help="Wipe outdir parquet first")
    return ap.parse_args()


def _convert_one(
    csv_path: Path,
    outdir: Path,
    *,
    name: str | None,
    normal_only: bool,
    max_rows: int | None,
) -> Path:
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"[Preprocess] reading {csv_path}")
    df = read_raw_can_csv(csv_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
        print(f"[Preprocess] capped to {max_rows:,} rows")

    eng = engineer_can_frames(df)
    if normal_only:
        before = len(eng)
        eng = eng[eng["is_anomaly"] == 0].copy()
        print(f"[Preprocess] normal-only: kept {len(eng):,}/{before:,}")

    stem = name or csv_path.stem
    out = outdir / f"{stem}.parquet"
    keep = ["timestamp", "can_id", "is_anomaly", *FEATURES]
    eng[keep].to_parquet(out, index=False)
    n_anom = int(eng["is_anomaly"].sum())
    print(
        f"[Preprocess] wrote {out.name}: {len(eng):,} rows "
        f"({n_anom:,} anomalous), features={len(FEATURES)}"
    )
    return out


def main() -> int:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for old in outdir.glob("*.parquet"):
            old.unlink()
            print(f"[Preprocess] removed {old.name}")

    jobs: list[tuple[Path, str | None]] = []
    if args.csv is not None:
        jobs.append((args.csv, args.name))
    else:
        jobs.append((NORMAL_CSV, args.name or "normal"))
        if args.include_attacks:
            for key, path in ATTACK_CSVS.items():
                jobs.append((path, key))

    written = []
    for csv_path, name in jobs:
        written.append(
            _convert_one(
                csv_path,
                outdir,
                name=name,
                normal_only=args.normal_only,
                max_rows=args.max_rows,
            )
        )

    print(f"[Preprocess] features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"[Preprocess] done → {outdir.resolve()} ({len(written)} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
