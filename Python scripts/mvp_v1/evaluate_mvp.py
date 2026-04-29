"""
================================================================================
  evaluate_mvp.py — minimum viable evaluation: GRU autoencoder vs Isolation Forest

  Purpose
  -------
  Prove (or disprove) that the GRU sequence-to-sequence autoencoder used by the
  federated IDS catches attacks that an Isolation Forest baseline misses, on
  the same vehicle telemetry, with the same calibration.

  Usage
  -----
      pip install scikit-learn pandas pyarrow
      python evaluate_mvp.py path/to/telemetry.parquet
      python evaluate_mvp.py path/to/folder_of_parquets/

  What it does
  ------------
  1. Loads parquet telemetry; auto-resolves column names; drops location_z.
  2. Splits sequentially 80% train / 20% test (no temporal leakage).
  3. Trains both detectors on the same training windows.
  4. Sets each detector's threshold at p99 × 1.5 of its own training errors.
  5. Injects four attack families into the test set: spike, drift, frequency,
     splice. Three of the four preserve marginal statistics, so an IID
     detector should miss them — that is the test.
  6. Prints AUROC, F1, per-attack detection rates, and inference latency.
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

# Reuse the production GRU exactly — what we evaluate is what runs on the Pi.
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
# Parquet loading (column auto-detection, drops location_z + timestamp)
# ──────────────────────────────────────────────────────────────────────────────

FEATURES = ("speed_kmh", "battery_level", "throttle", "brake",
            "steering", "gear", "location_x", "location_y")
assert len(FEATURES) == N_FEATURES

# Permissive aliases so different CARLA loggers all work.
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
    """Load one file or every .parquet under a directory; return (N, 8) array."""
    p = Path(path)
    files = sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
    if not files:
        sys.exit(f"No parquet files at {path}")

    dfs = [pd.read_parquet(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"[Data] loaded {len(df)} rows from {len(files)} file(s)")

    # Resolve aliases case-insensitively.
    cols_lower = {c.lower(): c for c in df.columns}
    chosen: dict[str, str] = {}
    for canonical in FEATURES:
        for alias in ALIASES[canonical]:
            if alias.lower() in cols_lower:
                chosen[canonical] = cols_lower[alias.lower()]
                break
        else:
            sys.exit(f"Missing required feature {canonical!r}.  "
                     f"Available columns: {list(df.columns)}")

    print(f"[Data] using columns: " +
          ", ".join(f"{k}←{v}" for k, v in chosen.items()))
    print(f"[Data] dropped: " +
          ", ".join(c for c in df.columns if c not in chosen.values()))

    arr = df[[chosen[c] for c in FEATURES]].to_numpy(dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def to_windows(trace: np.ndarray, mu=None, sd=None):
    """Z-score, then slide. Same as DataCollector.get_windows."""
    if mu is None:
        mu = trace.mean(axis=0)
        sd = np.where(trace.std(axis=0) > 1e-6, trace.std(axis=0), 1.0)
    norm = ((trace - mu) / sd).astype(np.float32)
    out  = np.stack([norm[i:i + WINDOW_SIZE].flatten()
                     for i in range(len(norm) - WINDOW_SIZE + 1)])
    return out, mu, sd


# ──────────────────────────────────────────────────────────────────────────────
# Detector wrappers — identical interface so the harness can treat them alike
# ──────────────────────────────────────────────────────────────────────────────

class IForestDetector:
    """Baseline IDS: treats every 160-D window as an IID feature vector."""
    def __init__(self):
        self.model     = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
        self.threshold = float("inf")

    def fit(self, windows):
        self.model.fit(windows)

    def score(self, w):
        # sklearn: high score = normal. Negate so high = anomalous (matches GRU).
        return -float(self.model.score_samples(w.reshape(1, -1))[0])

    def calibrate(self, windows):
        errs = np.array([self.score(w) for w in windows[:300]])
        self.threshold = float(np.percentile(errs, ANOMALY_PERCENTILE)
                               * ANOMALY_SAFETY_MULT)


class GRUDetector:
    """Wraps the production GRU autoencoder with the same .score/.calibrate API."""
    def __init__(self):
        self.model     = GRUAutoencoder(seed=42)
        self.threshold = float("inf")

    def fit(self, windows, epochs=20, batch_size=32):
        for ep in range(epochs):
            perm   = np.random.permutation(len(windows))
            losses = [self.model.train_step(windows[perm[i:i + batch_size]])
                      for i in range(0, len(windows), batch_size)
                      if len(windows[perm[i:i + batch_size]]) > 0]
            if (ep + 1) % 5 == 0:
                print(f"  GRU epoch {ep+1}/{epochs}  loss={np.mean(losses):.5f}")

    def score(self, w):
        return float(self.model.reconstruction_error(w))

    def calibrate(self, windows):
        errs = np.array([self.score(w) for w in windows[:300]])
        self.threshold = float(np.percentile(errs, ANOMALY_PERCENTILE)
                               * ANOMALY_SAFETY_MULT)


# ──────────────────────────────────────────────────────────────────────────────
# Attack injectors — each targets a different IID-detector blind spot
# ──────────────────────────────────────────────────────────────────────────────
# spike     : single huge outlier  → both should catch (sanity check)
# drift     : slow ramp on speed   → marginal stats preserved, IF blind
# frequency : high-freq steering   → marginal stats preserved, IF blind
# splice    : mid-window cutover   → marginal stats preserved, IF blind

def attack_spike(w, rng):
    s = w.reshape(WINDOW_SIZE, N_FEATURES).copy()
    s[rng.integers(WINDOW_SIZE), rng.integers(N_FEATURES)] += 6.0
    return s.flatten()

def attack_drift(w, rng):
    s = w.reshape(WINDOW_SIZE, N_FEATURES).copy()
    half = WINDOW_SIZE // 2
    s[half:, 0] += np.linspace(0, 1.6, WINDOW_SIZE - half, dtype=np.float32)
    return s.flatten()

def attack_frequency(w, rng):
    s = w.reshape(WINDOW_SIZE, N_FEATURES).copy()
    s[:, 4] = np.sin(np.pi * np.arange(WINDOW_SIZE, dtype=np.float32))   # steering
    return s.flatten()

def attack_splice(w_a, w_b, rng):
    a = w_a.reshape(WINDOW_SIZE, N_FEATURES).copy()
    b = w_b.reshape(WINDOW_SIZE, N_FEATURES)
    a[WINDOW_SIZE // 2:] = b[WINDOW_SIZE // 2:]
    return a.flatten()

ATTACKS = {"spike": attack_spike, "drift": attack_drift,
           "frequency": attack_frequency, "splice": attack_splice}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(detector, name, test_windows, labels, kinds):
    """Score every window, time each call, return a results dict."""
    scores, lat = [], []
    for w in test_windows:
        t0 = time.perf_counter()
        scores.append(detector.score(w))
        lat.append((time.perf_counter() - t0) * 1e6)
    scores = np.array(scores)
    preds  = (scores > detector.threshold).astype(int)
    per_attack = {k: float(preds[(kinds == k) & (labels == 1)].mean())
                  for k in ATTACKS}
    return {
        "name":      name,
        "auroc":     float(roc_auc_score(labels, scores)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "fp_rate":   float(preds[labels == 0].mean()),
        "per_attack": per_attack,
        "lat_p50":   float(np.percentile(lat, 50)),
        "lat_p99":   float(np.percentile(lat, 99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", help="parquet file or directory of parquet files")
    ap.add_argument("--train-ratio", type=float, default=0.80,
                    help="fraction of frames used to train both detectors (default 0.80)")
    ap.add_argument("--attack-rate", type=float, default=0.30,
                    help="fraction of test windows that get attacks injected")
    args = ap.parse_args()

    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Load and split sequentially (no shuffling — would leak future into past).
    trace = load_parquet(args.parquet)
    cut   = int(len(trace) * args.train_ratio)
    train_trace, test_trace = trace[:cut], trace[cut:]
    print(f"[Split] train={len(train_trace)} frames   test={len(test_trace)} frames")

    train_w, mu, sd  = to_windows(train_trace)
    test_w_clean, *_ = to_windows(test_trace, mu, sd)
    if len(train_w) < 50 or len(test_w_clean) < 50:
        sys.exit("Not enough data — need a longer parquet trace.")

    # Build attacked test set.
    n_atk = int(len(test_w_clean) * args.attack_rate)
    idxs  = rng.choice(len(test_w_clean), n_atk, replace=False)
    test_w = test_w_clean.copy()
    labels = np.zeros(len(test_w), dtype=int)
    kinds  = np.array(["benign"] * len(test_w), dtype=object)
    families = list(ATTACKS)
    for j, i in enumerate(idxs):
        k = families[j % 4]
        if k == "splice":
            donor = test_w_clean[(i + len(test_w) // 2) % len(test_w)]
            test_w[i] = ATTACKS[k](test_w_clean[i], donor, rng)
        else:
            test_w[i] = ATTACKS[k](test_w_clean[i], rng)
        labels[i] = 1
        kinds[i]  = k
    print(f"[Attacks] injected {n_atk} attacks across {len(families)} families")

    # Train + calibrate both.
    print("\n[IForest] fitting...")
    iforest = IForestDetector()
    iforest.fit(train_w)
    iforest.calibrate(train_w)
    print(f"[IForest] threshold = {iforest.threshold:.4f}")

    print("\n[GRU] training...")
    gru = GRUDetector()
    gru.fit(train_w)
    gru.calibrate(train_w)
    print(f"[GRU] threshold = {gru.threshold:.4f}")

    # Score and report.
    print("\n[Eval] scoring test set with both detectors...")
    r_if  = evaluate(iforest, "IForest", test_w, labels, kinds)
    r_gru = evaluate(gru,     "GRU",     test_w, labels, kinds)

    print("\n" + "=" * 64)
    print(f"  {'metric':<22}{'IForest':>14}{'GRU':>14}{'Δ':>10}")
    print("-" * 64)
    for key, label in [("auroc", "AUROC"), ("f1", "F1"),
                       ("fp_rate", "False-pos rate"),
                       ("lat_p50", "Latency p50 (µs)"),
                       ("lat_p99", "Latency p99 (µs)")]:
        a, b = r_if[key], r_gru[key]
        print(f"  {label:<22}{a:>14.3f}{b:>14.3f}{b-a:>+10.3f}")
    print("\n  Per-attack detection rate:")
    for k in ATTACKS:
        a, b = r_if["per_attack"][k], r_gru["per_attack"][k]
        print(f"    {k:<20}{a:>14.3f}{b:>14.3f}{b-a:>+10.3f}")
    print("=" * 64)

    if r_gru["auroc"] > r_if["auroc"] + 0.05:
        print("\nVerdict: GRU clearly outperforms Isolation Forest.")
    elif r_gru["auroc"] > r_if["auroc"]:
        print("\nVerdict: GRU edges ahead — gain may not justify cost.")
    else:
        print("\nVerdict: IForest competitive; investigate why GRU isn't winning.")


if __name__ == "__main__":
    main()
