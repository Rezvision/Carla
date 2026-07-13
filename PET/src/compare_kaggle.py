"""
Run baseline + all three DP perturbation families on the Kaggle tabular dataset
and collect results — a transfer sanity check vs. the CARLA GRU pipeline.

Key differences from compare.py (CARLA):
- Dataset: synthetic_telemetry_data.csv, 1,970 rows, 50 vehicles, 32 features
- Model: MLPAutoencoder (tabular, no windowing) vs GRUAutoencoder (sequential)
- Split: row-level class-stratified (normal 70/15/15; failures to val/test only)
- Anomaly eval: real failure labels via compute_metrics_labeled, not synthetic attacks
- MI non-members: normal-only rows from test vehicles (not the full test set)
- No personalized DP section (vehicle-level granularity is the split unit here)

Usage (from repo root):
    python -m src.compare_kaggle \
        --data_path data/synthetic_telemetry_data.csv \
        --out_dir notebooks --epochs 30
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.evaluate import compute_metrics_labeled, membership_inference
from src.kaggle_data import (
    KAGGLE_FEATURES,
    load_kaggle,
    make_kaggle_dataset,
    split_kaggle_stratified,
)
from src.model import MLPAutoencoder
from src.perturbation import (
    apply_input_noise,
    apply_output_noise,
    apply_output_noise_last_layer,
    analytic_gaussian_epsilon,
    worst_case_sigma,
    output_last_layer_epsilon,
    output_noise_epsilon,
)
from src.train import train

# Same sweep grid as compare.py so results are side-by-side comparable.
INPUT_SIGMAS = [0.01, 0.05, 0.1, 0.3]
# Uniform per-feature σ for the Kaggle 32-feature set (no domain-specific weighting).
INPUT_SIGMA_PER_FEATURE = [0.05] * len(KAGGLE_FEATURES)
CLIP_RANGE = 3.0
OUTPUT_SIGMAS = [0.0001, 0.001, 0.005, 0.01]
DPSGD_CLIPS = [0.5, 1.0, 5.0]
DPSGD_NMS = [1.0, 2.0]
DELTA = 1e-8  # << 1/(10n) for the train split size (see METHODS.md)


def _fresh_model():
    return MLPAutoencoder(input_size=len(KAGGLE_FEATURES), hidden_sizes=(16, 8))


def _save_partial_results(results, out_dir):
    """Write accumulated rows to kaggle_results_table_partial.csv after every run."""
    df = pd.DataFrame(results)
    if "clip_norm" not in df.columns:
        df["clip_norm"] = float("nan")
    if "epsilon_final" not in df.columns:
        df["epsilon_final"] = float("nan")
    df.to_csv(os.path.join(out_dir, "kaggle_results_table_partial.csv"), index=False)


def run(data_path, out_dir, epochs=30, batch_size=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed BEFORE any model is constructed so sweeps are reproducible across runs.
    torch.manual_seed(42)

    models_dir = os.path.join(out_dir, "kaggle_models")
    figures_dir = os.path.join(out_dir, "kaggle_figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("Loading Kaggle telemetry data …")
    df = load_kaggle(data_path)
    train_df, val_df, test_df = split_kaggle_stratified(df, seed=42)

    n_val_anom = int(val_df["anomaly_label"].sum())
    n_test_anom = int(test_df["anomaly_label"].sum())
    print(f"  Train: {len(train_df)} normal rows  (0 anomalies by design)")
    print(f"  Val:   {len(val_df)} rows, {n_val_anom} anomalies")
    print(f"  Test:  {len(test_df)} rows, {n_test_anom} anomalies")
    if n_test_anom < 5:
        print(f"  WARNING: only {n_test_anom} anomalies in test set — AUROC will be high-variance")

    train_ds, train_labels, scaler = make_kaggle_dataset(train_df, fit_scaler=True)
    val_ds, val_labels, _ = make_kaggle_dataset(val_df, scaler=scaler)
    test_ds, test_labels, _ = make_kaggle_dataset(test_df, scaler=scaler)

    # MI non-members = only normal test rows (anomalies would bias reconstruction errors).
    test_normal_mask = test_labels == 0
    test_normal_ds = TensorDataset(test_ds.tensors[0][torch.tensor(test_normal_mask)])

    results = []

    def _eval(model):
        """IDS detection metrics (real labels, val threshold) + MI privacy metrics."""
        m = compute_metrics_labeled(model, test_ds, test_labels,
                                    val_dataset=val_ds, val_labels=val_labels, device=device)
        m.update(membership_inference(model, train_ds, test_normal_ds, device=device))
        return m

    # ------------------------------------------------------------ baseline
    print("\n=== Baseline ===")
    baseline_model = _fresh_model()
    baseline_model = train(
        baseline_model, train_ds, val_ds,
        save_path=os.path.join(models_dir, "baseline.pth"),
        epochs=epochs, batch_size=batch_size, device=device,
    )
    metrics = _eval(baseline_model)
    results.append({"method": "Baseline", "sigma": None, "epsilon": float("inf"), **metrics})
    _save_partial_results(results, out_dir)

    # --------------------------------------------------- input perturbation
    print("\n=== Input Perturbation ===")
    sensitivity = 2 * CLIP_RANGE
    all_input_sigmas = [(s, str(s)) for s in INPUT_SIGMAS] + \
                       [(INPUT_SIGMA_PER_FEATURE, "per-feature")]
    for sigma, label in all_input_sigmas:
        print(f"  σ={label}")
        noisy_train_ds = apply_input_noise(train_ds, sigma=sigma, clip_range=CLIP_RANGE)
        model = _fresh_model()
        model = train(
            model, noisy_train_ds, val_ds,
            save_path=os.path.join(models_dir, f"input_sigma{label}.pth"),
            epochs=epochs, batch_size=batch_size, device=device,
        )
        eps = analytic_gaussian_epsilon(worst_case_sigma(sigma), sensitivity=sensitivity, delta=DELTA)
        metrics = _eval(model)
        results.append({"method": "Input", "sigma": label, "epsilon": eps, **metrics})
        _save_partial_results(results, out_dir)

    # -------------------------------------------------- output perturbation (full model)
    print("\n=== Output Perturbation (full model) ===")
    for sigma in OUTPUT_SIGMAS:
        print(f"  σ={sigma}")
        noisy_model = apply_output_noise(baseline_model, sigma=sigma)
        eps = output_noise_epsilon(baseline_model, sigma=sigma, delta=DELTA)
        metrics = _eval(noisy_model)
        results.append({"method": "Output", "sigma": sigma, "epsilon": eps, **metrics})
        _save_partial_results(results, out_dir)

    # ----------------------------------- output perturbation (last layer only)
    print("\n=== Output Perturbation (last-layer only) ===")
    for sigma in OUTPUT_SIGMAS:
        print(f"  σ={sigma}")
        noisy_model = apply_output_noise_last_layer(baseline_model, sigma=sigma)
        eps = output_last_layer_epsilon(baseline_model, sigma=sigma, delta=DELTA)
        metrics = _eval(noisy_model)
        results.append({"method": "Output (last-layer)", "sigma": sigma, "epsilon": eps, **metrics})
        _save_partial_results(results, out_dir)

    # ------------------------------------------------------------ DP-SGD
    print("\n=== DP-SGD (Opacus) ===")
    try:
        from src.train_dpsgd import train_dpsgd
    except ImportError:
        print("  Opacus not installed — skipping DP-SGD. Run: pip install opacus")
        train_dpsgd = None

    if train_dpsgd is not None:
        for clip in DPSGD_CLIPS:
            for nm in DPSGD_NMS:
                print(f"  clip={clip}  noise_multiplier={nm}")
                model = _fresh_model()
                model, eps, eps_final = train_dpsgd(
                    model, train_ds, val_ds,
                    save_path=os.path.join(models_dir, f"dpsgd_clip{clip}_nm{nm}.pth"),
                    noise_multiplier=nm, max_grad_norm=clip, delta=DELTA,
                    epochs=epochs, batch_size=batch_size, device=device,
                )
                metrics = _eval(model)
                results.append({
                    "method": "DP-SGD (fixed-clip baseline)",
                    "sigma": nm, "clip_norm": clip, "epsilon": eps, "epsilon_final": eps_final,
                    **metrics,
                })
                _save_partial_results(results, out_dir)

    # ------------------------------------------------------------ table
    df_out = pd.DataFrame(results)
    if "clip_norm" not in df_out.columns:
        df_out["clip_norm"] = float("nan")
    if "epsilon_final" not in df_out.columns:
        df_out["epsilon_final"] = float("nan")
    df_out = df_out[["method", "sigma", "clip_norm", "epsilon", "epsilon_final",
                      "auroc", "f1", "fpr_at_95tpr", "mean_mse_normal",
                      "mi_auc"]]
    df_out.columns = ["Method", "σ / noise_mult", "clip_norm", "ε", "ε (final epoch)",
                      "AUROC", "F1", "FPR@95TPR", "MSE(normal)", "MI-AUC"]
    table_path = os.path.join(out_dir, "kaggle_results_table.csv")
    df_out.to_csv(table_path, index=False)
    print(f"\n{df_out.to_string(index=False)}")
    print(f"\nTable saved to {table_path}")

    baseline_auroc = df_out.loc[df_out["Method"] == "Baseline", "AUROC"].values[0]
    baseline_mi = df_out.loc[df_out["Method"] == "Baseline", "MI-AUC"].values[0]
    _plot_epsilon(df_out, figures_dir, baseline_auroc=baseline_auroc)
    _plot_mi(df_out, figures_dir, baseline_auroc=baseline_auroc, baseline_mi=baseline_mi)

    return df_out


def _plot_epsilon(df, figures_dir, baseline_auroc):
    """Two-panel ε-based plot (mirrors compare.py layout)."""
    fig, (ax_local, ax_central) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "Privacy–Utility Tradeoff: MLP Autoencoder IDS (Kaggle Telemetry)\n"
        "(ε not comparable across panels — left=local/parameter DP, right=central DP)",
        fontsize=11,
    )

    for ax in (ax_local, ax_central):
        ax.axhline(baseline_auroc, color="black", linestyle="--", linewidth=1,
                   label=f"Baseline AUROC={baseline_auroc:.3f}")
        ax.set_ylabel("IDS AUROC (real labels)")
        ax.grid(True, which="both", alpha=0.3)

    local_styles = {
        "Input":               ("steelblue",  "o", "-"),
        "Output":              ("darkorange", "s", "-"),
        "Output (last-layer)": ("sienna",     "D", "--"),
    }
    for method, (color, marker, ls) in local_styles.items():
        sub = df[df["Method"] == method].copy()
        sub = sub[sub["ε"].notna() & (sub["ε"] < 1e10)].sort_values("ε")
        if sub.empty:
            continue
        ax_local.plot(sub["ε"], sub["AUROC"], marker=marker, linestyle=ls,
                      color=color, label=method)
    ax_local.set_xscale("log")
    ax_local.set_xlabel("ε (local/parameter DP, log scale)")
    ax_local.set_title("Input & Output perturbation")
    ax_local.legend(fontsize=8)

    dpsgd = df[df["Method"] == "DP-SGD (fixed-clip baseline)"].copy()
    dpsgd = dpsgd[dpsgd["ε"].notna()].copy()
    clip_markers = {0.5: "o", 1.0: "s", 5.0: "D"}
    for clip_val, marker in clip_markers.items():
        sub = dpsgd[dpsgd["clip_norm"] == clip_val].sort_values("ε")
        if sub.empty:
            continue
        ax_central.plot(sub["ε"], sub["AUROC"], marker=marker, linestyle="-",
                        color="seagreen", label=f"clip C={clip_val}")
    ax_central.set_xscale("log")
    ax_central.set_xlabel("ε (central DP, Opacus RDP accountant, log scale)")
    ax_central.set_title("DP-SGD (fixed-clip baseline)")
    ax_central.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(figures_dir, "kaggle_privacy_utility_epsilon.png")
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close(fig)


def _plot_mi(df, figures_dir, baseline_auroc, baseline_mi):
    """Single-panel MI-AUC privacy axis (mirrors compare.py layout)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(baseline_auroc, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(baseline_mi, color="black", linestyle=":", linewidth=1, alpha=0.6,
               label=f"Baseline (AUROC={baseline_auroc:.3f}, MI-AUC={baseline_mi:.3f})")

    style_map = {
        "Input":                        ("steelblue",  "o", "-"),
        "Output":                       ("darkorange", "s", "-"),
        "Output (last-layer)":          ("sienna",     "D", "--"),
        "DP-SGD (fixed-clip baseline)": ("seagreen",   "^", "-"),
    }
    for method, (color, marker, ls) in style_map.items():
        sub = df[df["Method"] == method].copy()
        sub = sub[sub["MI-AUC"].notna()].sort_values("MI-AUC")
        if sub.empty:
            continue
        ax.plot(sub["MI-AUC"], sub["AUROC"], marker=marker, linestyle=ls,
                color=color, label=method, zorder=3)

    ax.set_xlabel("MI Attack AUC  (0.5 = private, 1.0 = fully exposed)", fontsize=10)
    ax.set_ylabel("IDS AUROC (real labels)", fontsize=10)
    ax.set_title("Privacy–Utility Tradeoff: MI-AUC axis (Kaggle MLP)\n"
                 "(comparable across local DP, central DP)")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "kaggle_privacy_utility_mi.png")
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    # Windows consoles and redirected output default to cp1252, which cannot encode
    # the σ/ε characters used in progress prints and table headers — the 2026-07
    # definitive sweep crashed on exactly this straight after the baseline finished.
    # Force UTF-8 (replacing anything unencodable) before any such print can run.
    import sys
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/synthetic_telemetry_data.csv")
    parser.add_argument("--out_dir", default="notebooks")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    run(args.data_path, args.out_dir, args.epochs, args.batch_size)
