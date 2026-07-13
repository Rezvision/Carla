#!/usr/bin/env python3
"""
evaluate_vae.py — evaluate the trained LSTM-VAE with the paper's three detection heads.

Loads a saved model bundle, builds a benign test set from held-out parquet, injects
synthetic attacks, and reports the metrics the paper selected for imbalanced anomaly
detection (Sec. 4.1.1 / 4.3): PR AUC and macro F1 — plus recall / FPR at the
calibrated threshold, for each of:

    1. reconstruction error      (encoder + decoder)
    2. latent distance           (encoder-only, BallTree nearest neighbour)
    3. latent clustering          (encoder-only, Isolation Forest)

Run from Python scripts/vae_v1/:

    python evaluate_vae.py ../../Data/new_data_processed --bundle checkpoints --name vae
    python evaluate_vae.py ../../Data/new_data_processed/vehicle_1_20260321_145906.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
except ImportError:
    sys.exit("Install deps:  pip install scikit-learn pandas pyarrow")

from vae_model import LSTMVAE, FEATURES, N_FEATURES, WINDOW_SIZE, STRIDE
from train_vae import ALIASES, load_parquet, make_windows


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic attacks — operate on z-scored windows (WINDOW_SIZE, N_FEATURES);
# magnitudes are expressed in standard deviations.
# ──────────────────────────────────────────────────────────────────────────────

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
    # Flood: a burst pins several channels to a constant extreme (loss of variance).
    s = w.copy()
    start = rng.integers(0, WINDOW_SIZE - WINDOW_SIZE // 4)
    s[start:start + WINDOW_SIZE // 4, :] = magnitude
    return s

def attack_spoof(w, rng, magnitude=5.0):
    # Constant offset on a single spoofed signal channel.
    s = w.copy()
    ch = rng.integers(N_FEATURES)
    s[:, ch] += magnitude
    return s

def attack_replay(w, w_other, rng):
    # Stale/replayed frame: overwrite the whole window with an earlier one.
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


def build_attack_set(benign: np.ndarray, rng, attack_rate: float):
    """Return (windows, labels, kinds): a mix of benign + injected-attack windows."""
    n_attack = int(len(benign) * attack_rate)
    out_w, out_lbl, out_kind = [benign], [np.zeros(len(benign), int)], [np.array(["benign"] * len(benign))]

    per_kind = max(1, n_attack // len(ATTACKS))
    for name, fn in SINGLE_ATTACKS.items():
        idx = rng.integers(0, len(benign), per_kind)
        w = np.stack([fn(benign[i], rng) for i in idx])
        out_w.append(w); out_lbl.append(np.ones(per_kind, int)); out_kind.append(np.array([name] * per_kind))
    for name, fn in PAIR_ATTACKS.items():
        idx = rng.integers(0, len(benign), per_kind)
        idx2 = rng.integers(0, len(benign), per_kind)
        w = np.stack([fn(benign[i], benign[j], rng) for i, j in zip(idx, idx2)])
        out_w.append(w); out_lbl.append(np.ones(per_kind, int)); out_kind.append(np.array([name] * per_kind))

    return (np.concatenate(out_w).astype(np.float32),
            np.concatenate(out_lbl),
            np.concatenate(out_kind))


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def report_head(name, scores, pred, labels, kinds):
    prevalence = float(labels.mean())
    pr_auc = average_precision_score(labels, scores)
    auroc = roc_auc_score(labels, scores)
    macro_f1 = f1_score(labels, pred, average="macro", zero_division=0)
    recall = float(pred[labels == 1].mean())
    fpr = float(pred[labels == 0].mean())

    print(f"\n  == {name} ==")
    print(f"    PR AUC        : {pr_auc:.3f}   (random baseline = {prevalence:.3f})")
    print(f"    macro F1      : {macro_f1:.3f}")
    print(f"    AUROC         : {auroc:.3f}")
    print(f"    recall / FPR  : {recall:.3f} / {fpr:.3f}   (@ calibrated threshold)")
    print(f"    per-attack PR AUC vs benign:")
    benign = scores[labels == 0]
    for k in ATTACKS:
        m = (kinds == k) & (labels == 1)
        if m.any():
            y = np.concatenate([np.zeros(len(benign)), np.ones(int(m.sum()))])
            s = np.concatenate([benign, scores[m]])
            print(f"      {k:<10} {average_precision_score(y, s):.3f}")
    return {"head": name, "pr_auc": pr_auc, "macro_f1": macro_f1,
            "auroc": auroc, "recall": recall, "fpr": fpr}


def main():
    ap = argparse.ArgumentParser(description="Evaluate LSTM-VAE (paper detection heads)")
    ap.add_argument("data", type=Path, help="parquet file or directory (held-out normal data)")
    ap.add_argument("--bundle", type=Path, default=Path("checkpoints"))
    ap.add_argument("--name", default="vae")
    ap.add_argument("--attack-rate", type=float, default=0.30)
    ap.add_argument("--max-windows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Load model bundle + scaler ─────────────────────────────────────────────
    model = LSTMVAE.load(str(args.bundle), name=args.name)
    scaler = np.load(args.bundle / f"{args.name}_scaler.npz")
    mu, sd = scaler["mean"], scaler["std"]
    print(f"[Model] loaded bundle from {args.bundle.resolve()}")
    print(f"[Model] recon_threshold={model.recon_threshold:.6f}  "
          f"dist_threshold={model.dist_threshold:.6f}")

    # ── Benign test set (standardised with the training scaler) ────────────────
    trace = load_parquet(args.data)
    norm = ((trace - mu) / sd).astype(np.float32)
    benign = make_windows(norm, STRIDE)
    if len(benign) == 0:
        sys.exit("No benign windows built from the given data.")
    if len(benign) > args.max_windows:
        idx = np.linspace(0, len(benign) - 1, args.max_windows, dtype=int)
        benign = benign[idx]
    print(f"[Test] benign windows: {len(benign):,}")

    windows, labels, kinds = build_attack_set(benign, rng, args.attack_rate)
    print(f"[Test] total windows:  {len(windows):,}  "
          f"(attacks={int(labels.sum()):,}, prevalence={labels.mean():.3f})")

    # ── Score all three heads ──────────────────────────────────────────────────
    recon_scores = model.reconstruction_error(windows)
    recon_pred = (recon_scores > model.recon_threshold).astype(int)

    dist_scores = model.latent_distance(windows)
    dist_pred = (dist_scores > model.dist_threshold).astype(int)

    clus_scores = model.latent_cluster_score(windows)
    clus_pred = model.latent_cluster_predict(windows)

    results = [
        report_head("Reconstruction (encoder+decoder)", recon_scores, recon_pred, labels, kinds),
        report_head("Latent distance (encoder only)", dist_scores, dist_pred, labels, kinds),
        report_head("Latent clustering (encoder only)", clus_scores, clus_pred, labels, kinds),
    ]

    print("\n  Summary (paper metrics)")
    print(f"    {'head':<34}{'PR AUC':>9}{'macroF1':>9}{'AUROC':>8}{'recall':>8}{'FPR':>7}")
    for r in results:
        print(f"    {r['head']:<34}{r['pr_auc']:>9.3f}{r['macro_f1']:>9.3f}"
              f"{r['auroc']:>8.3f}{r['recall']:>8.3f}{r['fpr']:>7.3f}")


if __name__ == "__main__":
    main()
