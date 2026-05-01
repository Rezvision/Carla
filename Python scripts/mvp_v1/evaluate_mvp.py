"""
================================================================================
  evaluate_mvp.py — GRU autoencoder vs Isolation Forest on CARLA telemetry

  Run:
      pip install scikit-learn pandas pyarrow
      python evaluate_mvp.py path/to/telemetry.parquet
      python evaluate_mvp.py path/to/folder_of_parquets/
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import f1_score, roc_auc_score
except ImportError:
    sys.exit("Install deps:  pip install scikit-learn pandas pyarrow")

try:
    from fed_client_jax import (
        ANOMALY_PERCENTILE,
        ANOMALY_SAFETY_MULT,
        N_FEATURES,
        WINDOW_SIZE,
        GRUAutoencoder,
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


def load_parquet(path: str) -> np.ndarray:
    p = Path(path)
    files = sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
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

    print(f"[Data] using columns:  " +
          ", ".join(f"{k}←{v}" for k, v in chosen.items()))
    print(f"[Data] dropped:        " +
          ", ".join(c for c in df.columns if c not in chosen.values()))

    arr = df[[chosen[c] for c in FEATURES]].to_numpy(dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def to_windows(trace: np.ndarray, mu=None, sd=None):
    if mu is None:
        mu = trace.mean(axis=0)
        sd = np.where(trace.std(axis=0) > 1e-6, trace.std(axis=0), 1.0)
    norm = ((trace - mu) / sd).astype(np.float32)
    out = np.stack([norm[i:i + WINDOW_SIZE].flatten()
                    for i in range(len(norm) - WINDOW_SIZE + 1)])
    return out, mu, sd


# ──────────────────────────────────────────────────────────────────────────────
# Detectors
# ──────────────────────────────────────────────────────────────────────────────

class IForestDetector:
    def __init__(self):
        self.model = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
        self.threshold = float("inf")

    def fit(self, windows):
        self.model.fit(windows)

    def score(self, w):
        return -float(self.model.score_samples(w.reshape(1, -1))[0])

    def score_batch(self, windows):
        # Vectorised — much faster than the per-row .score() Python loop.
        return -self.model.score_samples(windows)


class GRUDetector:
    def __init__(self):
        self.model = GRUAutoencoder(seed=42)
        self.threshold = float("inf")

    def fit(self, windows, epochs=20, batch_size=32):
        for ep in range(epochs):
            perm = np.random.permutation(len(windows))
            losses = []
            for i in range(0, len(windows), batch_size):
                b = windows[perm[i:i + batch_size]]
                if len(b):
                    losses.append(self.model.train_step(b))
            if (ep + 1) % 5 == 0:
                print(f"  GRU epoch {ep+1:2d}/{epochs}  loss={np.mean(losses):.5f}")

    def score(self, w):
        return float(self.model.reconstruction_error(w))


def calibrate_uniform(detector, train_windows, n_samples=2000, kind="batch"):
    """
    Sample calibration windows UNIFORMLY across the training set
    (not the first N — those are typically degenerate launch frames).
    """
    n = len(train_windows)
    take = min(n_samples, n)
    idx = np.linspace(0, n - 1, take, dtype=int)
    samples = train_windows[idx]
    if kind == "batch":
        errs = detector.score_batch(samples)
    else:
        errs = np.array([detector.score(w) for w in samples])
    p99 = float(np.percentile(errs, ANOMALY_PERCENTILE))
    detector.threshold = p99 * ANOMALY_SAFETY_MULT
    return errs


# ──────────────────────────────────────────────────────────────────────────────
# Attacks (operate on z-scored windows; magnitudes are in std-deviations)
# ──────────────────────────────────────────────────────────────────────────────

def attack_spike(w, rng, magnitude=8.0):
    s = w.reshape(WINDOW_SIZE, N_FEATURES).copy()
    s[rng.integers(WINDOW_SIZE), rng.integers(N_FEATURES)] += magnitude
    return s.flatten()

def attack_drift(w, rng, magnitude=3.0):
    s = w.reshape(WINDOW_SIZE, N_FEATURES).copy()
    half = WINDOW_SIZE // 2
    s[half:, 0] += np.linspace(0, magnitude, WINDOW_SIZE - half, dtype=np.float32)
    return s.flatten()

def attack_frequency(w, rng, amplitude=2.0):
    s = w.reshape(WINDOW_SIZE, N_FEATURES).copy()
    s[:, 4] = amplitude * np.sin(np.pi * np.arange(WINDOW_SIZE, dtype=np.float32))
    return s.flatten()

def attack_splice(w_a, w_b, rng):
    a = w_a.reshape(WINDOW_SIZE, N_FEATURES).copy()
    b = w_b.reshape(WINDOW_SIZE, N_FEATURES)
    a[WINDOW_SIZE // 2:] = b[WINDOW_SIZE // 2:]
    return a.flatten()

ATTACKS = {"spike": attack_spike, "drift": attack_drift,
           "frequency": attack_frequency, "splice": attack_splice}


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def score_dist(scores: np.ndarray) -> str:
    return (f"mean={scores.mean():.3f}  "
            f"p50={np.percentile(scores, 50):.3f}  "
            f"p95={np.percentile(scores, 95):.3f}  "
            f"p99={np.percentile(scores, 99):.3f}  "
            f"max={scores.max():.3f}")


def best_threshold(scores: np.ndarray, labels: np.ndarray):
    """
    Sweep thresholds, return the one that maximises F1.
    Decouples model quality from threshold-calibration quality.
    """
    candidates = np.unique(np.percentile(scores, np.linspace(50, 99.9, 200)))
    best = (float(np.median(scores)), 0.0, 0.0, 0.0)
    for t in candidates:
        pred = (scores > t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec  = tp / (tp + fn)
        f1   = 2 * prec * rec / (prec + rec)
        if f1 > best[1]:
            best = (float(t), f1, prec, rec)
    return best


def report(name, scores, labels, kinds, threshold, latencies_us):
    print(f"\n  ── {name} ─────────────────────────────────────────────────────")
    print(f"    calibrated threshold:        {threshold:.4f}")
    print(f"    benign score dist:           {score_dist(scores[labels == 0])}")
    for k in ATTACKS:
        m = (kinds == k) & (labels == 1)
        if m.any():
            print(f"    {k:<10} score dist:       {score_dist(scores[m])}")

    # Per-attack AUROC (each family vs benign) — threshold-independent.
    # Tells us whether the score *ranks* attacks above benign at all.
    print(f"    per-attack AUROC vs benign:")
    benign_scores = scores[labels == 0]
    for k in ATTACKS:
        m = (kinds == k) & (labels == 1)
        if m.any():
            y = np.concatenate([np.zeros(len(benign_scores)), np.ones(m.sum())])
            s = np.concatenate([benign_scores, scores[m]])
            auc = roc_auc_score(y, s)
            print(f"      {k:<10}                  {auc:.3f}")

    overall_auc = roc_auc_score(labels, scores)
    pred = (scores > threshold).astype(int)
    f1c = f1_score(labels, pred, zero_division=0)
    fpc = float(pred[labels == 0].mean())
    rec = float(pred[labels == 1].mean())

    bt, bf1, bp, br = best_threshold(scores, labels)
    print(f"    overall AUROC:               {overall_auc:.3f}")
    print(f"    @ calibrated threshold:      F1={f1c:.3f}  recall={rec:.3f}  FP={fpc:.3f}")
    print(f"    @ optimal threshold {bt:>7.3f}: F1={bf1:.3f}  precision={bp:.3f}  recall={br:.3f}")
    print(f"    latency p50 / p99 (µs):      {np.percentile(latencies_us, 50):.1f}  /  "
          f"{np.percentile(latencies_us, 99):.1f}")

    return {"auroc": overall_auc, "f1_calibrated": f1c, "f1_optimal": bf1,
            "lat_p50": float(np.percentile(latencies_us, 50)),
            "lat_p99": float(np.percentile(latencies_us, 99))}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", help="parquet file or directory")
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--attack-rate", type=float, default=0.30)
    ap.add_argument("--max-test-windows", type=int, default=20_000,
                    help="cap test set size — IForest scoring is O(n·trees) per call")
    args = ap.parse_args()

    np.random.seed(42)
    rng = np.random.default_rng(42)

    # ── Load + sequential split ──────────────────────────────────────────
    trace = load_parquet(args.parquet)
    cut = int(len(trace) * args.train_ratio)
    train_trace, test_trace = trace[:cut], trace[cut:]
    print(f"[Split] train={len(train_trace):,}  test={len(test_trace):,}  "
          f"(sequential — no leakage)")

    train_w, mu, sd = to_windows(train_trace)
    test_w_clean, *_ = to_windows(test_trace, mu, sd)
    print(f"[Windows] train={len(train_w):,}  test={len(test_w_clean):,}")

    # Subsample test set if huge — keeps IForest scoring tractable.
    if len(test_w_clean) > args.max_test_windows:
        idx = np.linspace(0, len(test_w_clean) - 1, args.max_test_windows, dtype=int)
        test_w_clean = test_w_clean[idx]
        print(f"[Windows] test subsampled to {len(test_w_clean):,} (uniform across trace)")

    # ── Inject attacks ───────────────────────────────────────────────────
    n_atk = int(len(test_w_clean) * args.attack_rate)
    atk_idx = rng.choice(len(test_w_clean), n_atk, replace=False)
    test_w = test_w_clean.copy()
    labels = np.zeros(len(test_w), dtype=int)
    kinds = np.array(["benign"] * len(test_w), dtype=object)
    families = list(ATTACKS)
    for j, i in enumerate(atk_idx):
        k = families[j % len(families)]
        if k == "splice":
            donor = test_w_clean[(i + len(test_w) // 2) % len(test_w)]
            test_w[i] = ATTACKS[k](test_w_clean[i], donor, rng)
        else:
            test_w[i] = ATTACKS[k](test_w_clean[i], rng)
        labels[i] = 1
        kinds[i] = k
    print(f"[Attacks] " + ", ".join(f"{k}={int((kinds == k).sum())}" for k in families))

    # ── Train ────────────────────────────────────────────────────────────
    print("\n[IForest] fitting...")
    t0 = time.perf_counter()
    iforest = IForestDetector()
    iforest.fit(train_w)
    print(f"[IForest] fit done ({time.perf_counter() - t0:.1f}s)")

    print("\n[GRU] training (20 epochs)...")
    t0 = time.perf_counter()
    gru = GRUDetector()
    gru.fit(train_w)
    print(f"[GRU] training done ({time.perf_counter() - t0:.1f}s)")

    # ── Calibrate (uniform sample, NOT the first 300 frames) ─────────────
    print("\n[Calibration] sampling 2000 windows uniformly across training set...")
    if_calib = calibrate_uniform(iforest, train_w, kind="batch")
    print(f"[IForest] calib err dist:  {score_dist(if_calib)}")
    print(f"[IForest] threshold:       {iforest.threshold:.4f}  "
          f"(p{ANOMALY_PERCENTILE} × {ANOMALY_SAFETY_MULT})")

    gru_calib = calibrate_uniform(gru, train_w, kind="loop")
    print(f"[GRU]     calib err dist:  {score_dist(gru_calib)}")
    print(f"[GRU]     threshold:       {gru.threshold:.4f}  "
          f"(p{ANOMALY_PERCENTILE} × {ANOMALY_SAFETY_MULT})")

    # ── Score test set ───────────────────────────────────────────────────
    print("\n[Eval] scoring test set...")
    t0 = time.perf_counter()
    if_scores = iforest.score_batch(test_w)
    if_total = (time.perf_counter() - t0) * 1e6
    if_lat = np.full(len(test_w), if_total / len(test_w))   # batch — equal share
    print(f"[IForest] {len(test_w):,} windows in {if_total/1e6:.1f}s "
          f"({if_total/len(test_w):.1f} µs/window amortised)")

    print("[GRU] per-window scoring...")
    gru_scores = np.empty(len(test_w))
    gru_lat = np.empty(len(test_w))
    for i, w in enumerate(test_w):
        t0 = time.perf_counter()
        gru_scores[i] = gru.score(w)
        gru_lat[i] = (time.perf_counter() - t0) * 1e6

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    r_if = report("IForest", if_scores, labels, kinds, iforest.threshold, if_lat)
    r_gru = report("GRU", gru_scores, labels, kinds, gru.threshold, gru_lat)

    print("\n" + "=" * 72)
    print("  HEAD-TO-HEAD")
    print("=" * 72)
    fmt = "  {:<28}{:>14}{:>14}{:>10}"
    print(fmt.format("metric", "IForest", "GRU", "Δ"))
    for k, lbl in [("auroc",         "AUROC (overall)"),
                   ("f1_calibrated", "F1 @ calibrated thr"),
                   ("f1_optimal",    "F1 @ optimal thr"),
                   ("lat_p50",       "Latency p50 (µs)"),
                   ("lat_p99",       "Latency p99 (µs)")]:
        a, b = r_if[k], r_gru[k]
        print(fmt.format(lbl, f"{a:.3f}", f"{b:.3f}", f"{b-a:+.3f}"))

    print("\nReading the results:")
    print("  • AUROC and F1@optimal show what each detector CAN do (model quality).")
    print("  • F1@calibrated shows what they DO at the deployed threshold.")
    print("  • If F1@optimal >> F1@calibrated, threshold is the bottleneck, not the model.")


if __name__ == "__main__":
    main()
