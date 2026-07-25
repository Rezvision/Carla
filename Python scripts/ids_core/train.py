#!/usr/bin/env python3
"""
Unified trainer for CARLA + Kaggle + raw CAN.

    cd "Python scripts"
    python -m experiments.train --dataset carla  --model gru     --epochs 20
    python -m experiments.train --dataset kaggle --model fsmn    --epochs 50
    python -m datasets.can.preprocess
    python -m experiments.train --dataset can    --model can_gru --epochs 20
    python -m experiments.train --dataset can    --model can_vae --epochs 50 --objective recon

Checkpoints default to experiments/checkpoints/{dataset}/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python train.py` when cwd is ids_core/ or Python scripts/
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ids_core.models import MODEL_NAMES  # noqa: E402
from ids_core.profiles import DATASET_NAMES  # noqa: E402
from ids_core.trainer import TrainConfig, train  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("data", type=Path, nargs="?", default=None,
                    help="parquet file/dir (default: profile data dir)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--safety-mult", type=float, default=None,
                    help="GRU threshold multiplier (default: profile)")
    ap.add_argument("--k", type=float, default=2.5)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--objective", choices=("recon", "latent"), default="recon")
    ap.add_argument("--ae-latent", type=int, default=None)
    ap.add_argument("--noise-std", type=float, default=0.1)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model_kwargs: dict = {}
    fit_kwargs: dict = {"lr": args.lr}
    if args.safety_mult is not None:
        model_kwargs["safety_mult"] = args.safety_mult
    if args.model == "fsmn":
        model_kwargs["k"] = args.k
        model_kwargs["l1"] = args.l1
        fit_kwargs["k"] = args.k
        fit_kwargs["l1"] = args.l1
    elif args.model in ("vae", "can_vae"):
        model_kwargs["objective"] = args.objective
    elif args.model == "tet":
        model_kwargs["k"] = args.k
        model_kwargs["noise_std"] = args.noise_std
        if args.ae_latent is not None:
            model_kwargs["ae_latent"] = args.ae_latent
        fit_kwargs["k"] = args.k
        fit_kwargs["noise_std"] = args.noise_std

    train(TrainConfig(
        dataset=args.dataset,
        model=args.model,
        data=args.data,
        outdir=args.outdir,
        name=args.name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        window=args.window,
        val_ratio=args.val_ratio,
        max_windows=args.max_windows,
        seed=args.seed,
        model_kwargs=model_kwargs,
        fit_kwargs=fit_kwargs,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
