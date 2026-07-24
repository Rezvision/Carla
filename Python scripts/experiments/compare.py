#!/usr/bin/env python3
"""
Dispatch fair model comparison by dataset.

    python -m experiments.compare --dataset carla  ../Data/carla/processed/new
    python -m experiments.compare --dataset kaggle ../Data/kaggle/processed
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=("carla", "kaggle"))
    args, rest = ap.parse_known_args(argv)

    script = _HERE / ("compare_carla.py" if args.dataset == "carla" else "compare_kaggle.py")
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
