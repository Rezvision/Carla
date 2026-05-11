"""
Filter parquet files: keep only rows where the battery column is not 0.

Usage:
    python filter_battery.py <input_folder> [--column battery] [--output processed_vehicle_logs]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def filter_file(src: Path, dst: Path, column: str) -> tuple[int, int]:
    """Filter one parquet file. Returns (rows_in, rows_out)."""
    df = pd.read_parquet(src)

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in {src.name}. "
            f"Available columns: {list(df.columns)}"
        )

    rows_in = len(df)
    filtered = df[df[column] != 0]
    rows_out = len(filtered)

    dst.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(dst, index=False)
    return rows_in, rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path, help="Folder containing .parquet files")
    parser.add_argument(
        "--column",
        default="battery",
        help="Column to filter on (default: battery)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed_vehicle_logs"),
        help="Output folder (default: processed_vehicle_logs)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories",
    )
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        print(f"Error: '{args.input_folder}' is not a directory", file=sys.stderr)
        return 1

    pattern = "**/*.parquet" if args.recursive else "*.parquet"
    files = sorted(args.input_folder.glob(pattern))

    if not files:
        print(f"No .parquet files found in {args.input_folder}")
        return 0

    total_in = total_out = 0
    failures = 0

    for src in files:
        rel = src.relative_to(args.input_folder) if args.recursive else src.name
        dst = args.output / rel
        try:
            rows_in, rows_out = filter_file(src, dst, args.column)
        except Exception as e:
            print(f"  [SKIP] {rel}: {e}", file=sys.stderr)
            failures += 1
            continue

        dropped = rows_in - rows_out
        total_in += rows_in
        total_out += rows_out
        print(f"  {rel}: kept {rows_out}/{rows_in} (dropped {dropped})")

    print(
        f"\nDone. {len(files) - failures}/{len(files)} files processed. "
        f"Kept {total_out}/{total_in} rows total. Output: {args.output}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
