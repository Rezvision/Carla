#!/usr/bin/env python3
"""
Evaluate a Kaggle-trained model using dataset failure labels (not synthetic CAN attacks).

Supports:
  - gru / fsmn / vae / tet checkpoints under experiments/checkpoints/kaggle/

    python experiments/evaluate_kaggle.py ../Data/kaggle/processed --model gru --name gru_kaggle
    python experiments/evaluate_kaggle.py ../Data/kaggle/processed --model fsmn --name fsmn_kaggle
    python experiments/evaluate_kaggle.py ../Data/kaggle/processed --model tet --name tet_kaggle
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SCRIPTS = HERE.parent
if str(_SCRIPTS) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPTS))
from datasets.kaggle.config import CHECKPOINT_DIR, FEATURES, N_FEATURES, PROCESSED_DIR, STRIDE, WINDOW_SIZE
from datasets.kaggle.data_utils import flatten_windows, load_dataset


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores > threshold).astype(np.int8)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auroc = float(roc_auc_score(y_true, scores)) if y_true.min() != y_true.max() else float("nan")
        ap = float(average_precision_score(y_true, scores)) if y_true.min() != y_true.max() else float("nan")
    except Exception:
        auroc, ap = float("nan"), float("nan")

    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "auroc": auroc,
        "average_precision": ap,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "score_mean_normal": float(scores[y_true == 0].mean()) if (y_true == 0).any() else float("nan"),
        "score_mean_anomaly": float(scores[y_true == 1].mean()) if (y_true == 1).any() else float("nan"),
    }


def eval_gru(data: Path, outdir: Path, name: str, stride: int, window: int) -> dict:
    from models.gru import GRUAutoencoder

    ds = load_dataset(data, normal_only=False, stride=stride, window=window, fit_scaler_on="normal")
    model = GRUAutoencoder(n_features=N_FEATURES, window_size=window, checkpoint_dir=outdir)
    model.restore_checkpoint(name)
    flat = flatten_windows(ds["windows"])
    scores = model.reconstruction_errors(flat)
    return _metrics(ds["labels"], scores, model.threshold)


def eval_fsmn(data: Path, outdir: Path, name: str, stride: int, window: int) -> dict:
    from models.fsmn import FSMNAE

    ds = load_dataset(data, normal_only=False, stride=stride, window=window, fit_scaler_on="normal")
    model = FSMNAE.load(str(outdir), name=name)
    scores = np.asarray(model.reconstruction_error(ds["windows"]), dtype=np.float32)
    thr = float(model.recon_threshold)
    return _metrics(ds["labels"], scores, thr)


def eval_vae(data: Path, outdir: Path, name: str, stride: int, window: int) -> dict:
    from models.vae import LSTMVAE

    ds = load_dataset(data, normal_only=False, stride=stride, window=window, fit_scaler_on="normal")
    model = LSTMVAE.load(str(outdir), name=name)
    scores = np.asarray(model.reconstruction_error(ds["windows"]), dtype=np.float32)
    thr = float(model.recon_threshold)
    return _metrics(ds["labels"], scores, thr)


def eval_tet(data: Path, outdir: Path, name: str, stride: int, window: int) -> dict:
    from models.tet import TETAE

    ds = load_dataset(data, normal_only=False, stride=stride, window=window, fit_scaler_on="normal")
    model = TETAE.load(str(outdir), name=name)
    scores = np.asarray(
        model.reconstruction_error(ds["windows"], ds["timestamps"]),
        dtype=np.float32,
    )
    thr = float(model.recon_threshold)
    return _metrics(ds["labels"], scores, thr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, nargs="?", default=PROCESSED_DIR)
    ap.add_argument("--model", choices=("gru", "fsmn", "vae", "tet"), default="gru")
    ap.add_argument("--outdir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--name", default=None, help="checkpoint/bundle name")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--window", type=int, default=WINDOW_SIZE)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    defaults = {
        "gru": "gru_kaggle",
        "fsmn": "fsmn_kaggle",
        "vae": "vae_kaggle",
        "tet": "tet_kaggle",
    }
    name = args.name or defaults[args.model]
    print(f"[Eval] model={args.model}  name={name}  features={list(FEATURES)}")

    dispatch = {
        "gru": eval_gru,
        "fsmn": eval_fsmn,
        "vae": eval_vae,
        "tet": eval_tet,
    }
    metrics = dispatch[args.model](args.data, args.outdir, name, args.stride, args.window)

    print(json.dumps(metrics, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))
        print(f"[Eval] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
