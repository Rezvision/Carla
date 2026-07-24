#!/usr/bin/env python3
"""
compare_models.py — fair comparison of GRU / VAE / FSMN-AE / TET on Kaggle data.

Uses the same processed pool, window/stride, and *dataset failure labels*
(not synthetic CAN attacks). Writes figures under experiments/figures/kaggle/:

  headline_comparison.png
  detection_quality.png
  operating_point.png
  accuracy_vs_cost.png
  footprint.png
  feature_heatmap.png
  metrics.csv

    python -m experiments.compare --dataset kaggle ../Data/kaggle/processed
    python experiments/compare_kaggle.py ../Data/kaggle/processed --max-windows 10000 --latency-n 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_MPLDIR = _HERE / "figures" / "kaggle" / ".mplconfig"
_MPLDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from datasets.kaggle.config import (
    CHECKPOINT_DIR,
    FEATURES,
    N_FEATURES,
    PROCESSED_DIR,
    STRIDE,
    WINDOW_SIZE,
)
from datasets.kaggle.data_utils import (
    anomaly_mask,
    feature_matrix,
    flatten_windows,
    load_vehicle_frames,
    make_windows,
    timestamp_vector,
)

MODEL_COLORS = {
    "GRU-AE": "#1b9e77",
    "VAE": "#d95f02",
    "FSMN-AE": "#7570b3",
    "TET": "#e7298a",
}


def _add_sys_path(subdir: str):
    p = str(_HERE.parent / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)


def windows_with_scaler(
    path: Path,
    mu: np.ndarray,
    sd: np.ndarray,
    *,
    stride: int,
    window: int,
    max_windows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (windows, timestamps, labels) using a model-specific scaler."""
    frames = load_vehicle_frames(path)
    all_w, all_t, all_y = [], [], []
    for df in frames:
        feats = feature_matrix(df)
        labels = anomaly_mask(df)
        ts = timestamp_vector(df)
        norm = ((feats - mu) / sd).astype(np.float32)
        w, y, t = make_windows(
            norm, stride=stride, window=window, labels=labels, timestamps=ts,
        )
        if len(w):
            all_w.append(w)
            all_t.append(t)
            all_y.append(y)
    if not all_w:
        raise SystemExit("No windows produced for comparison.")
    windows = np.concatenate(all_w, axis=0)
    timestamps = np.concatenate(all_t, axis=0)
    labels = np.concatenate(all_y, axis=0)

    if max_windows and len(windows) > max_windows:
        rng = np.random.default_rng(seed)
        # Stratified-ish: keep all anomalies, subsample normals to fill budget
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]
        keep_pos = pos if len(pos) <= max_windows else rng.choice(pos, max_windows, replace=False)
        remain = max_windows - len(keep_pos)
        keep_neg = rng.choice(neg, min(remain, len(neg)), replace=False) if remain > 0 else np.array([], dtype=int)
        idx = np.sort(np.concatenate([keep_pos, keep_neg]))
        windows, timestamps, labels = windows[idx], timestamps[idx], labels[idx]
    return windows, timestamps, labels


def metrics_row(name, scores, pred, labels, *, n_params, size_kb, lat_p50, lat_p99):
    return {
        "model": name,
        "PR AUC": float(average_precision_score(labels, scores)) if labels.min() != labels.max() else float("nan"),
        "AUROC": float(roc_auc_score(labels, scores)) if labels.min() != labels.max() else float("nan"),
        "macro F1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, pred)) if labels.min() != labels.max() else 0.0,
        "recall": float(pred[labels == 1].mean()) if labels.any() else 0.0,
        "FPR": float(pred[labels == 0].mean()) if (labels == 0).any() else 0.0,
        "params": int(n_params),
        "size_kib": float(size_kb),
        "latency_p50_us": float(lat_p50),
        "latency_p99_us": float(lat_p99),
        "prevalence": float(labels.mean()),
        "n_windows": int(len(labels)),
        "n_pos": int(labels.sum()),
    }


def streaming_latency(score_fn, windows, n: int, warmup: int = 8) -> tuple[float, float]:
    n = min(n, len(windows))
    idx = np.linspace(0, len(windows) - 1, n, dtype=int)
    for i in idx[:warmup]:
        score_fn(windows[i:i + 1])
    lat = np.empty(n, dtype=np.float64)
    for k, i in enumerate(idx):
        t0 = time.perf_counter()
        score_fn(windows[i:i + 1])
        lat[k] = (time.perf_counter() - t0) * 1e6
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 99))


# ── Per-model evaluators ──────────────────────────────────────────────────────

def eval_gru(data: Path, args) -> dict:
    from models.gru import GRUAutoencoder

    name = args.gru_name
    bundle = Path(args.bundle)
    ckpt = bundle / f"{name}.npz"
    scaler_path = bundle / f"{name}_scaler.npz"
    if not ckpt.is_file():
        raise SystemExit(
            f"GRU checkpoint missing: {ckpt}\n"
            f"Train: python train_gru.py {data} --epochs 20 --name {name}"
        )

    sc = np.load(scaler_path)
    mu, sd = sc["mean"], sc["std"]
    windows, timestamps, labels = windows_with_scaler(
        data, mu, sd, stride=args.stride, window=args.window,
        max_windows=args.max_windows, seed=args.seed,
    )
    _ = timestamps

    meta = {}
    meta_path = bundle / f"{name}_meta.json"
    if meta_path.is_file():
        import json
        meta = json.loads(meta_path.read_text())
    window = int(meta.get("window_size", args.window))
    n_features = int(meta.get("n_features", N_FEATURES))

    model = GRUAutoencoder(
        n_features=n_features, window_size=window,
        seed=args.seed, checkpoint_dir=bundle,
    )
    model.restore_checkpoint(name)
    flat = flatten_windows(windows)
    scores = model.reconstruction_errors(flat).astype(np.float64)
    pred = (scores > model.threshold).astype(int)

    def _score_batch(batch_3d):
        return model.reconstruction_error(batch_3d[0].reshape(-1))

    lat_p50, lat_p99 = streaming_latency(_score_batch, windows, args.latency_n)
    n_params = int(sum(np.prod(np.array(v).shape) for v in model.params.values()))
    size_kb = n_params * 4 / 1024.0
    print(f"[GRU] windows={len(windows):,}  params={n_params:,}  "
          f"θ={model.threshold:.4f}  p50={lat_p50:.1f}µs")
    return metrics_row("GRU-AE", scores, pred, labels,
                       n_params=n_params, size_kb=size_kb,
                       lat_p50=lat_p50, lat_p99=lat_p99)


def eval_vae(data: Path, args) -> dict:
    # vae from models.vae
    from models.vae.model import LSTMVAE  # type: ignore

    name = args.vae_name
    bundle = Path(args.bundle)
    if not (bundle / f"{name}_meta.json").is_file():
        raise SystemExit(
            f"VAE bundle missing under {bundle} ({name}_*).\n"
            f"Train: python train_vae.py {data} --epochs 50 --name {name}"
        )

    model = LSTMVAE.load(str(bundle), name=name)
    sc = np.load(bundle / f"{name}_scaler.npz")
    windows, timestamps, labels = windows_with_scaler(
        data, sc["mean"], sc["std"], stride=args.stride, window=args.window,
        max_windows=args.max_windows, seed=args.seed,
    )
    _ = timestamps
    scores = np.asarray(model.reconstruction_error(windows), dtype=np.float64)
    pred = (scores > model.recon_threshold).astype(int)
    lat_p50, lat_p99 = streaming_latency(
        lambda b: model.reconstruction_error(b), windows, args.latency_n)
    n_params = model.num_params() if hasattr(model, "num_params") else -1
    if n_params < 0:
        import jax
        n_params = int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(model.params)))
    size_kb = n_params * 4 / 1024.0
    print(f"[VAE] windows={len(windows):,}  params={n_params:,}  "
          f"θ={model.recon_threshold:.4f}  p50={lat_p50:.1f}µs")
    return metrics_row("VAE", scores, pred, labels,
                       n_params=n_params, size_kb=size_kb,
                       lat_p50=lat_p50, lat_p99=lat_p99)


def eval_fsmn(data: Path, args) -> dict:
    # fsmn from models.fsmn
    from models.fsmn.model import FSMNAE  # type: ignore

    name = args.fsmn_name
    bundle = Path(args.bundle)
    if not (bundle / f"{name}_meta.json").is_file():
        raise SystemExit(
            f"FSMN bundle missing under {bundle} ({name}_*).\n"
            f"Train: python train_fsmn.py {data} --epochs 50 --name {name}"
        )

    model = FSMNAE.load(str(bundle), name=name)
    sc = np.load(bundle / f"{name}_scaler.npz")
    windows, timestamps, labels = windows_with_scaler(
        data, sc["mean"], sc["std"], stride=args.stride, window=args.window,
        max_windows=args.max_windows, seed=args.seed,
    )
    _ = timestamps
    scores = np.asarray(model.reconstruction_error(windows), dtype=np.float64)
    pred = (scores > model.recon_threshold).astype(int)
    lat_p50, lat_p99 = streaming_latency(
        lambda b: model.reconstruction_error(b), windows, args.latency_n)
    n_params = model.num_params()
    size_kb = model.approx_size_kb()
    print(f"[FSMN] windows={len(windows):,}  params={n_params:,}  "
          f"θ={model.recon_threshold:.4f}  p50={lat_p50:.1f}µs")
    return metrics_row("FSMN-AE", scores, pred, labels,
                       n_params=n_params, size_kb=size_kb,
                       lat_p50=lat_p50, lat_p99=lat_p99)


def eval_tet(data: Path, args) -> dict:
    # tet from models.tet
    from models.tet.model import TETAE  # type: ignore

    name = args.tet_name
    bundle = Path(args.bundle)
    if not (bundle / f"{name}_meta.json").is_file():
        raise SystemExit(
            f"TET bundle missing under {bundle} ({name}_*).\n"
            f"Train: python train_tet.py {data} --epochs 50 --name {name}"
        )

    model = TETAE.load(str(bundle), name=name)
    sc = np.load(bundle / f"{name}_scaler.npz")
    windows, timestamps, labels = windows_with_scaler(
        data, sc["mean"], sc["std"], stride=args.stride, window=args.window,
        max_windows=args.max_windows, seed=args.seed,
    )
    scores = np.asarray(model.reconstruction_error(windows, timestamps), dtype=np.float64)
    pred = (scores > model.recon_threshold).astype(int)

    n = min(args.latency_n, len(windows))
    idx = np.linspace(0, len(windows) - 1, n, dtype=int)
    for i in idx[:8]:
        model.reconstruction_error(windows[i:i + 1], timestamps[i:i + 1])
    lat = np.empty(n, dtype=np.float64)
    for k, i in enumerate(idx):
        t0 = time.perf_counter()
        model.reconstruction_error(windows[i:i + 1], timestamps[i:i + 1])
        lat[k] = (time.perf_counter() - t0) * 1e6
    lat_p50, lat_p99 = float(np.percentile(lat, 50)), float(np.percentile(lat, 99))

    n_params = model.num_params()
    size_kb = model.approx_size_kb()
    print(f"[TET] windows={len(windows):,}  params={n_params:,}  "
          f"θ={model.recon_threshold:.4f}  p50={lat_p50:.1f}µs")
    return metrics_row("TET", scores, pred, labels,
                       n_params=n_params, size_kb=size_kb,
                       lat_p50=lat_p50, lat_p99=lat_p99)


# ── Plotting (same layout as Python scripts/compare_models.py) ────────────────

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })


def plot_detection_quality(df: pd.DataFrame, out: Path):
    metrics = ["PR AUC", "AUROC", "macro F1", "MCC"]
    models = df["model"].tolist()
    x = np.arange(len(models))
    n_m = len(metrics)
    width = 0.8 / n_m
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, m in enumerate(metrics):
        vals = df[m].to_numpy()
        bars = ax.bar(x + offsets[i], vals, width, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    prev = float(df["prevalence"].iloc[0])
    ax.axhline(prev, color="#666666", ls="--", lw=1,
               label=f"PR-AUC random ({prev:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ymin = min(0.0, float(df[metrics].min().min()) - 0.05)
    ax.set_ylim(ymin, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Detection quality (Kaggle failure labels)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_operating_point(df: pd.DataFrame, out: Path):
    models = df["model"].tolist()
    x = np.arange(len(models))
    metrics = [("recall", "recall @ θ", "#4c78a8"),
               ("FPR", "FPR @ θ", "#f58518"),
               ("MCC", "MCC @ θ", "#54a24b")]
    n_m = len(metrics)
    width = 0.8 / n_m
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (col, label, color) in enumerate(metrics):
        vals = df[col].to_numpy()
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ymin = min(0.0, float(df[["recall", "FPR", "MCC"]].min().min()) - 0.05)
    ax.set_ylim(ymin, 1.12)
    ax.set_ylabel("Score / rate")
    ax.set_title("Operating point at calibrated threshold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_accuracy_vs_cost(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for _, row in df.iterrows():
        ax.scatter(row["latency_p50_us"], row["PR AUC"],
                   s=float(np.clip(row["params"] / 200.0, 80, 900)),
                   color=MODEL_COLORS.get(row["model"], "#333"),
                   alpha=0.85, edgecolors="black", linewidths=0.6,
                   label=row["model"], zorder=3)
        ax.annotate(row["model"],
                    (row["latency_p50_us"], row["PR AUC"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Streaming latency p50 (µs)")
    ax.set_ylabel("PR AUC")
    ax.set_title("Accuracy vs cost (marker size ∝ parameters)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9, title="model")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_footprint(df: pd.DataFrame, out: Path):
    models = df["model"].tolist()
    x = np.arange(len(models))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = [MODEL_COLORS.get(m, "#333") for m in models]
    ax0.bar(x, df["params"], color=colors)
    ax0.set_yscale("log")
    ax0.set_xticks(x)
    ax0.set_xticklabels(models, rotation=15, ha="right")
    ax0.set_ylabel("Parameters (log)")
    ax0.set_title("Model size")
    for i, v in enumerate(df["params"]):
        ax0.text(i, v * 1.15, f"{v:,}", ha="center", fontsize=8)

    width = 0.35
    ax1.bar(x - width / 2, df["latency_p50_us"], width, label="p50", color="#4c78a8")
    ax1.bar(x + width / 2, df["latency_p99_us"], width, label="p99", color="#e45756")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.set_ylabel("Latency (µs)")
    ax1.set_title("Streaming inference latency")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame, out: Path):
    models = df["model"].tolist()
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, "#333") for m in models]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    quality = ["PR AUC", "AUROC", "macro F1", "MCC"]
    n_m = len(quality)
    width = 0.8 / n_m
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width
    for i, m in enumerate(quality):
        ax.bar(x + offsets[i], df[m], width, label=m)
    prev = float(df["prevalence"].iloc[0])
    ax.axhline(prev, color="#666", ls="--", lw=1, label=f"PR baseline ({prev:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ymin = min(0.0, float(df[quality].min().min()) - 0.05)
    ax.set_ylim(ymin, 1.12)
    ax.set_title("Detection quality")
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    ax = axes[0, 1]
    op = [("recall", "recall", "#4c78a8"),
          ("FPR", "FPR", "#f58518"),
          ("MCC", "MCC", "#54a24b")]
    n_m = len(op)
    width = 0.8 / n_m
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width
    for i, (col, label, color) in enumerate(op):
        ax.bar(x + offsets[i], df[col], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ymin = min(0.0, float(df[["recall", "FPR", "MCC"]].min().min()) - 0.05)
    ax.set_ylim(ymin, 1.12)
    ax.set_title("Operating point @ θ")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for _, row in df.iterrows():
        ax.scatter(row["latency_p50_us"], row["PR AUC"],
                   s=float(np.clip(row["params"] / 200.0, 80, 900)),
                   color=MODEL_COLORS.get(row["model"], "#333"),
                   alpha=0.85, edgecolors="black", linewidths=0.5)
        ax.annotate(row["model"], (row["latency_p50_us"], row["PR AUC"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("Latency p50 (µs)")
    ax.set_ylabel("PR AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs latency")

    ax = axes[1, 1]
    ax.bar(x, df["params"], color=colors)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Parameters (log)")
    ax.set_title("Model footprint")

    fig.suptitle("Kaggle IDS comparison — reconstruction detectors",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_feature_heatmap(path: Path, out: Path, *, max_rows: int = 200_000):
    frames = load_vehicle_frames(path)
    sample = np.concatenate([feature_matrix(df) for df in frames], axis=0)
    n = len(sample)
    if n > max_rows:
        idx = np.linspace(0, n - 1, max_rows, dtype=int)
        sample = sample[idx]
        note = f" (n={max_rows:,} of {n:,})"
    else:
        note = f" (n={n:,})"

    sd = sample.std(axis=0)
    corr = np.corrcoef(sample.T)
    corr = np.nan_to_num(corr, nan=0.0)
    labels = list(FEATURES)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_plot = np.ma.array(corr, mask=mask)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr_plot, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Kaggle feature correlation heatmap{note}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if mask[i, j]:
                continue
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.55 else "#222",
                    fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    dead = [FEATURES[i] for i, s in enumerate(sd) if s < 1e-6]
    if dead:
        print(f"[Heatmap] near-constant features: {dead}")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(corr, index=labels, columns=labels).to_csv(
        out.with_name("feature_correlation.csv"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Kaggle GRU / VAE / FSMN / TET")
    ap.add_argument("data", type=Path, nargs="?", default=PROCESSED_DIR)
    ap.add_argument("--max-windows", type=int, default=10_000)
    ap.add_argument("--latency-n", type=int, default=400)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--window", type=int, default=WINDOW_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, default=_HERE / "figures" / "kaggle")
    ap.add_argument("--bundle", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--gru-name", default="gru_kaggle")
    ap.add_argument("--vae-name", default="vae_kaggle")
    ap.add_argument("--fsmn-name", default="fsmn_kaggle")
    ap.add_argument("--tet-name", default="tet_kaggle")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["gru", "vae", "fsmn", "tet"])
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data not found: {args.data}\nRun: python preprocess.py")

    args.outdir.mkdir(parents=True, exist_ok=True)
    _style()
    print(f"[Config] features={N_FEATURES} window={args.window} stride={args.stride} "
          f"max_windows={args.max_windows} latency_n={args.latency_n} seed={args.seed}")
    print(f"[Config] labels = dataset failure_type / is_anomaly (not synthetic CAN attacks)")

    print("\n=== Feature heatmap ===")
    plot_feature_heatmap(args.data, args.outdir / "feature_heatmap.png")
    print(f"[Saved] {args.outdir / 'feature_heatmap.png'}")

    rows = []
    evaluators = [
        ("gru", eval_gru),
        ("vae", eval_vae),
        ("fsmn", eval_fsmn),
        ("tet", eval_tet),
    ]
    for key, fn in evaluators:
        if key in args.skip:
            print(f"[Skip] {key}")
            continue
        print(f"\n=== Evaluating {key.upper()} ===")
        rows.append(fn(args.data, args))

    if not rows:
        print("[Warn] no models evaluated — heatmap only.")
        return 0

    df = pd.DataFrame(rows)
    csv_path = args.outdir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Table]\n{df.to_string(index=False, float_format=lambda v: f'{v:.3f}')}")
    print(f"\n[Saved] {csv_path}")

    plot_detection_quality(df, args.outdir / "detection_quality.png")
    plot_operating_point(df, args.outdir / "operating_point.png")
    plot_accuracy_vs_cost(df, args.outdir / "accuracy_vs_cost.png")
    plot_footprint(df, args.outdir / "footprint.png")
    plot_dashboard(df, args.outdir / "headline_comparison.png")
    print(f"[Saved] figures → {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
