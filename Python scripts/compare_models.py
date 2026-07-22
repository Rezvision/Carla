#!/usr/bin/env python3
"""
compare_models.py — fair headline comparison of MVP / VAE / FSMN-AE / TET.

Loads each trained reconstruction detector, evaluates on the same parquet pool,
same seed, same attack families, and writes:

  figures/headline_comparison.png   (2×2 dashboard)
  figures/detection_quality.png
  figures/operating_point.png
  figures/accuracy_vs_cost.png
  figures/footprint.png
  figures/feature_heatmap.png       (feature–feature correlation)
  figures/metrics.csv

Run from Python scripts/:

    python compare_models.py ../Data/new_data_processed
    python compare_models.py ../Data/new_data_processed --max-windows 10000 --latency-n 500
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Writable matplotlib cache (sandbox / CI friendly).
_MPLDIR = Path(__file__).resolve().parent / "figures" / ".mplconfig"
_MPLDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLDIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
WINDOW_SIZE = 20
STRIDE = 20
N_FEATURES = 8
FEATURES = (
    "speed_kmh", "battery_level", "throttle", "brake",
    "steering", "gear", "location_x", "location_y",
)
ALIASES = {
    "speed_kmh": ("speed_kmh", "speed", "velocity"),
    "battery_level": ("battery_level", "battery", "soc"),
    "throttle": ("throttle", "throttle_pct"),
    "brake": ("brake", "brake_pct"),
    "steering": ("steering", "steer", "steering_angle"),
    "gear": ("gear", "current_gear"),
    "location_x": ("location_x", "loc_x", "pos_x", "x"),
    "location_y": ("location_y", "loc_y", "pos_y", "y"),
}
TS_ALIASES = ("timestamp", "time", "ts", "datetime", "date_time")

# Same synthetic attacks as vae_v1 / fsmn_ae_v1 / tet_v1.
def attack_spike(w, rng, magnitude=8.0):
    s = w.copy()
    s[rng.integers(WINDOW_SIZE), rng.integers(N_FEATURES)] += magnitude
    return s

def attack_drift(w, rng, magnitude=3.0):
    s = w.copy()
    half = WINDOW_SIZE // 2
    s[half:, 0] += np.linspace(0, magnitude, WINDOW_SIZE - half, dtype=np.float32)
    return s

def attack_frequency(w, rng, amplitude=2.0):
    s = w.copy()
    s[:, 4] = amplitude * np.sin(np.pi * np.arange(WINDOW_SIZE, dtype=np.float32))
    return s

def attack_splice(w, w_other, rng):
    s = w.copy()
    s[WINDOW_SIZE // 2:] = w_other[WINDOW_SIZE // 2:]
    return s

def attack_fuzzing(w, rng, magnitude=4.0):
    s = w.copy()
    mask = rng.random(s.shape) < 0.15
    s[mask] += rng.normal(0, magnitude, size=int(mask.sum())).astype(np.float32)
    return s

def attack_dos(w, rng, magnitude=6.0):
    s = w.copy()
    start = rng.integers(0, WINDOW_SIZE - WINDOW_SIZE // 4)
    s[start:start + WINDOW_SIZE // 4, :] = magnitude
    return s

def attack_spoof(w, rng, magnitude=5.0):
    s = w.copy()
    ch = rng.integers(N_FEATURES)
    s[:, ch] += magnitude
    return s

def attack_replay(w, w_other, rng):
    return w_other.copy()


SINGLE_ATTACKS = {
    "dos": attack_dos,
    "fuzzing": attack_fuzzing,
    "spoof": attack_spoof,
    "spike": attack_spike,
    "drift": attack_drift,
    "frequency": attack_frequency,
}
PAIR_ATTACKS = {
    "replay": attack_replay,
    "splice": attack_splice,
}
ATTACKS = list(SINGLE_ATTACKS) + list(PAIR_ATTACKS)

MODEL_COLORS = {
    "MVP (GRU-AE)": "#1b9e77",
    "VAE": "#d95f02",
    "FSMN-AE": "#7570b3",
    "TET": "#e7298a",
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [N,F], timestamps [N] normalised to [0,1])."""
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not files:
        sys.exit(f"No parquet files at {path}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"[Data] loaded {len(df):,} rows from {len(files)} file(s)")

    cols_lower = {c.lower(): c for c in df.columns}
    chosen = {}
    for canonical in FEATURES:
        for alias in ALIASES[canonical]:
            if alias.lower() in cols_lower:
                chosen[canonical] = cols_lower[alias.lower()]
                break
        else:
            sys.exit(f"Missing feature {canonical!r}. Available: {list(df.columns)}")

    arr = df[[chosen[c] for c in FEATURES]].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    ts_col = next((cols_lower[a] for a in TS_ALIASES if a in cols_lower), None)
    if ts_col is not None:
        series = df[ts_col]
        if pd.api.types.is_numeric_dtype(series):
            raw = series.to_numpy(dtype=np.float64)
        else:
            parsed = pd.to_datetime(series, errors="coerce")
            raw = parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.float64)
            raw[pd.isna(parsed).to_numpy()] = np.nan
        tmin, tmax = float(np.nanmin(raw)), float(np.nanmax(raw))
        denom = (tmax - tmin) if tmax > tmin else 1.0
        ts = ((raw - tmin) / denom).astype(np.float32)
    else:
        n = len(arr)
        ts = (np.arange(n, dtype=np.float32) / max(n - 1, 1))
    return arr, np.nan_to_num(ts, nan=0.0, posinf=1.0, neginf=0.0)


def make_windows(norm: np.ndarray, ts: np.ndarray, stride: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    idxs = list(range(0, len(norm) - WINDOW_SIZE + 1, stride))
    if not idxs:
        return (np.zeros((0, WINDOW_SIZE, norm.shape[1]), dtype=np.float32),
                np.zeros((0, WINDOW_SIZE), dtype=np.float32))
    feat = np.stack([norm[i:i + WINDOW_SIZE] for i in idxs]).astype(np.float32)
    tsw = np.stack([ts[i:i + WINDOW_SIZE] for i in idxs]).astype(np.float32)
    return feat, tsw


def build_attack_set(benign: np.ndarray, benign_ts: np.ndarray, rng, attack_rate: float):
    n_attack = int(len(benign) * attack_rate)
    out_w = [benign]
    out_ts = [benign_ts]
    out_lbl = [np.zeros(len(benign), int)]
    out_kind = [np.array(["benign"] * len(benign))]

    per_kind = max(1, n_attack // len(ATTACKS))
    for name, fn in SINGLE_ATTACKS.items():
        idx = rng.integers(0, len(benign), per_kind)
        w = np.stack([fn(benign[i], rng) for i in idx])
        out_w.append(w)
        out_ts.append(benign_ts[idx])
        out_lbl.append(np.ones(per_kind, int))
        out_kind.append(np.array([name] * per_kind))
    for name, fn in PAIR_ATTACKS.items():
        idx = rng.integers(0, len(benign), per_kind)
        idx2 = rng.integers(0, len(benign), per_kind)
        w = np.stack([fn(benign[i], benign[j], rng) for i, j in zip(idx, idx2)])
        out_w.append(w)
        out_ts.append(benign_ts[idx])
        out_lbl.append(np.ones(per_kind, int))
        out_kind.append(np.array([name] * per_kind))

    return (np.concatenate(out_w).astype(np.float32),
            np.concatenate(out_ts).astype(np.float32),
            np.concatenate(out_lbl),
            np.concatenate(out_kind))


def metrics_row(name, scores, pred, labels, *, n_params, size_kb, lat_p50, lat_p99):
    return {
        "model": name,
        "PR AUC": float(average_precision_score(labels, scores)),
        "AUROC": float(roc_auc_score(labels, scores)),
        "macro F1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, pred)),
        "recall": float(pred[labels == 1].mean()) if labels.any() else 0.0,
        "FPR": float(pred[labels == 0].mean()) if (labels == 0).any() else 0.0,
        "params": int(n_params),
        "size_kib": float(size_kb),
        "latency_p50_us": float(lat_p50),
        "latency_p99_us": float(lat_p99),
        "prevalence": float(labels.mean()),
    }


def streaming_latency(score_fn, windows, n: int, warmup: int = 8) -> tuple[float, float]:
    """One window per call → p50 / p99 (µs)."""
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


def _add_sys_path(subdir: str):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Per-model evaluators ──────────────────────────────────────────────────────

def eval_mvp(trace, ts, args, rng) -> dict:
    _add_sys_path("mvp_v1")
    from fed_client_jax import GRUAutoencoder, CHECKPOINT_DIR  # type: ignore

    name = args.mvp_name
    ckpt = Path(CHECKPOINT_DIR) / f"{name}.npz"
    scaler_path = Path(CHECKPOINT_DIR) / f"{name}_scaler.npz"
    if not ckpt.exists():
        sys.exit(
            f"MVP checkpoint not found at {ckpt}.\n"
            f"Train first:  cd mvp_v1 && python train_central.py "
            f"{args.data} --stride 20 --epochs 20 --name {name}"
        )

    model = GRUAutoencoder(seed=args.seed)
    model.restore_checkpoint(name)
    scaler = np.load(scaler_path)
    mu, sd = scaler["mean"], scaler["std"]

    norm = ((trace - mu) / sd).astype(np.float32)
    benign, benign_ts = make_windows(norm, ts, STRIDE)
    if len(benign) > args.max_windows:
        idx = np.linspace(0, len(benign) - 1, args.max_windows, dtype=int)
        benign, benign_ts = benign[idx], benign_ts[idx]

    windows, _, labels, _ = build_attack_set(
        benign, benign_ts, rng, args.attack_rate)

    # MVP scores flat (T*F,) windows one at a time.
    flat = windows.reshape(len(windows), -1)
    scores = np.array([model.reconstruction_error(w) for w in flat], dtype=np.float64)
    pred = (scores > model.threshold).astype(int)

    def _score_batch(batch_3d):
        return model.reconstruction_error(batch_3d[0].reshape(-1))

    lat_p50, lat_p99 = streaming_latency(_score_batch, windows, args.latency_n)

    n_params = int(sum(np.prod(np.array(v).shape) for v in model.params.values()))
    size_kb = n_params * 4 / 1024.0
    print(f"[MVP] windows={len(windows):,}  params={n_params:,}  "
          f"θ={model.threshold:.4f}  p50={lat_p50:.1f}µs")
    return metrics_row("MVP (GRU-AE)", scores, pred, labels,
                       n_params=n_params, size_kb=size_kb,
                       lat_p50=lat_p50, lat_p99=lat_p99)


def eval_vae(trace, ts, args, rng) -> dict:
    _add_sys_path("vae_v1")
    from vae_model import LSTMVAE  # type: ignore

    bundle = ROOT / "vae_v1" / "checkpoints"
    model = LSTMVAE.load(str(bundle), name=args.vae_name)
    scaler = np.load(bundle / f"{args.vae_name}_scaler.npz")
    mu, sd = scaler["mean"], scaler["std"]

    norm = ((trace - mu) / sd).astype(np.float32)
    benign, benign_ts = make_windows(norm, ts, STRIDE)
    if len(benign) > args.max_windows:
        idx = np.linspace(0, len(benign) - 1, args.max_windows, dtype=int)
        benign, benign_ts = benign[idx], benign_ts[idx]

    windows, _, labels, _ = build_attack_set(
        benign, benign_ts, rng, args.attack_rate)
    scores = model.reconstruction_error(windows)
    pred = (scores > model.recon_threshold).astype(int)

    lat_p50, lat_p99 = streaming_latency(
        lambda b: model.reconstruction_error(b), windows, args.latency_n)

    n_params = model.num_params() if hasattr(model, "num_params") else -1
    if n_params < 0:
        import jax
        n_params = int(sum(np.prod(p.shape)
                           for p in jax.tree_util.tree_leaves(model.params)))
    size_kb = n_params * 4 / 1024.0
    print(f"[VAE] windows={len(windows):,}  params={n_params:,}  "
          f"θ={model.recon_threshold:.4f}  p50={lat_p50:.1f}µs")
    return metrics_row("VAE", scores, pred, labels,
                       n_params=n_params, size_kb=size_kb,
                       lat_p50=lat_p50, lat_p99=lat_p99)


def eval_fsmn(trace, ts, args, rng) -> dict:
    _add_sys_path("fsmn_ae_v1")
    from fsmn_model import FSMNAE  # type: ignore

    bundle = ROOT / "fsmn_ae_v1" / "checkpoints"
    model = FSMNAE.load(str(bundle), name=args.fsmn_name)
    scaler = np.load(bundle / f"{args.fsmn_name}_scaler.npz")
    mu, sd = scaler["mean"], scaler["std"]

    norm = ((trace - mu) / sd).astype(np.float32)
    benign, benign_ts = make_windows(norm, ts, STRIDE)
    if len(benign) > args.max_windows:
        idx = np.linspace(0, len(benign) - 1, args.max_windows, dtype=int)
        benign, benign_ts = benign[idx], benign_ts[idx]

    windows, _, labels, _ = build_attack_set(
        benign, benign_ts, rng, args.attack_rate)
    scores = model.reconstruction_error(windows)
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


def eval_tet(trace, ts, args, rng) -> dict:
    _add_sys_path("tet_v1")
    from tet_model import TETAE  # type: ignore

    bundle = ROOT / "tet_v1" / "checkpoints"
    model = TETAE.load(str(bundle), name=args.tet_name)
    scaler = np.load(bundle / f"{args.tet_name}_scaler.npz")
    mu, sd = scaler["mean"], scaler["std"]

    norm = ((trace - mu) / sd).astype(np.float32)
    benign, benign_ts = make_windows(norm, ts, STRIDE)
    if len(benign) > args.max_windows:
        idx = np.linspace(0, len(benign) - 1, args.max_windows, dtype=int)
        benign, benign_ts = benign[idx], benign_ts[idx]

    windows, timestamps, labels, _ = build_attack_set(
        benign, benign_ts, rng, args.attack_rate)
    scores = model.reconstruction_error(windows, timestamps)
    pred = (scores > model.recon_threshold).astype(int)

    # Streaming latency needs matching timestamps per window.
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


# ── Plotting ──────────────────────────────────────────────────────────────────

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
    ax.set_title("Detection quality (reconstruction head)")
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
    sizes = np.clip(df["params"].to_numpy(dtype=float) / 200.0, 80, 900)
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
    # unused sizes silenced
    del sizes
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
    """Combined 2×2 headline figure."""
    models = df["model"].tolist()
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, "#333") for m in models]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) detection quality
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

    # (0,1) operating point
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

    # (1,0) pareto
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

    # (1,1) footprint
    ax = axes[1, 1]
    ax.bar(x, df["params"], color=colors)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Parameters (log)")
    ax.set_title("Model footprint")

    fig.suptitle("IDS model comparison — reconstruction detectors",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir := out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    del outdir


def plot_feature_heatmap(trace: np.ndarray, out: Path, *, max_rows: int = 200_000):
    """
    Feature–feature Pearson correlation heatmap for the translated CAN signals.
    Subsamples uniformly if the trace is very large.
    """
    n = len(trace)
    if n > max_rows:
        idx = np.linspace(0, n - 1, max_rows, dtype=int)
        sample = trace[idx]
        note = f" (n={max_rows:,} of {n:,})"
    else:
        sample = trace
        note = f" (n={n:,})"

    # Drop constant / near-constant channels to avoid NaN correlations.
    sd = sample.std(axis=0)
    corr = np.corrcoef(sample.T)
    corr = np.nan_to_num(corr, nan=0.0)

    labels = list(FEATURES)
    # Lower triangle only (incl. diagonal) — correlation is symmetric.
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_plot = np.ma.array(corr, mask=mask)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr_plot, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(f"Feature correlation heatmap{note}")

    for i in range(len(labels)):
        for j in range(len(labels)):
            if mask[i, j]:
                continue
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.55 else "#222",
                    fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    # Flag near-constant channels in the title footnote via print.
    dead = [FEATURES[i] for i, s in enumerate(sd) if s < 1e-6]
    if dead:
        print(f"[Heatmap] near-constant features: {dead}")

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Also save the numeric matrix for reuse (full symmetric matrix).
    corr_df = pd.DataFrame(corr, index=labels, columns=labels)
    corr_df.to_csv(out.with_name("feature_correlation.csv"))
    return corr_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compare MVP / VAE / FSMN / TET")
    ap.add_argument("data", type=Path, help="parquet file or directory")
    ap.add_argument("--attack-rate", type=float, default=0.30)
    ap.add_argument("--max-windows", type=int, default=10_000)
    ap.add_argument("--latency-n", type=int, default=400,
                    help="windows used for streaming latency (p50/p99)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "figures")
    ap.add_argument("--mvp-name", default="central")
    ap.add_argument("--vae-name", default="vae")
    ap.add_argument("--fsmn-name", default="fsmn_ae")
    ap.add_argument("--tet-name", default="tet_ae")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["mvp", "vae", "fsmn", "tet"],
                    help="skip named models")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    _style()

    print(f"[Config] window={WINDOW_SIZE} stride={STRIDE} "
          f"attack_rate={args.attack_rate} max_windows={args.max_windows} "
          f"latency_n={args.latency_n} seed={args.seed}")

    trace, ts = load_parquet(args.data)
    print("\n=== Feature heatmap ===")
    plot_feature_heatmap(trace, args.outdir / "feature_heatmap.png")
    print(f"[Saved] {args.outdir / 'feature_heatmap.png'}")

    rows = []

    evaluators = [
        ("mvp", eval_mvp),
        ("vae", eval_vae),
        ("fsmn", eval_fsmn),
        ("tet", eval_tet),
    ]
    for key, fn in evaluators:
        if key in args.skip:
            print(f"[Skip] {key}")
            continue
        # Independent RNG stream per model so attack indices match given same seed,
        # while avoiding cross-model RNG consumption coupling.
        rng = np.random.default_rng(args.seed)
        print(f"\n=== Evaluating {key.upper()} ===")
        rows.append(fn(trace, ts, args, rng))

    if not rows:
        print("[Warn] no models evaluated — heatmap only.")
        print(f"[Saved] figures → {args.outdir.resolve()}")
        return

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


if __name__ == "__main__":
    main()
