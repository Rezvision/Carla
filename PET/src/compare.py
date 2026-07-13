"""
Run baseline + all three DP perturbation families and collect results.

Usage (from repo root):
    python -m src.compare --data_dir data/CARLA_processed --out_dir notebooks
"""
import argparse
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset

from src.data import (
    load_sessions_with_vehicle_ids, split_sessions, split_sessions_for_mi,
    session_groups_for_windows, make_dataset, FEATURES, WINDOW,
)
from src.model import GRUAutoencoder
from src.train import train
from src.perturbation import (
    apply_input_noise, analytic_gaussian_epsilon, worst_case_sigma,
    apply_output_noise, output_noise_epsilon,
    apply_output_noise_last_layer, output_last_layer_epsilon,
)
from src.evaluate import compute_metrics, membership_inference, balance_members, balanced_subsample_indices
from src.personalized import personalized_noise_map, apply_personalized_input_noise, per_vehicle_mi


# The 'different settings' each family is swept over. Each value trains/evaluates a
# separate model, so we get a privacy–utility CURVE rather than a single point.
INPUT_SIGMAS = [0.01, 0.05, 0.1, 0.3]
# Per-feature σ: least noise (0.05) on the fast-changing control signals (speed,
# throttle, brake, steering); moderate noise (0.1) on battery_level/gear (not
# explicitly fast-changing or positional); most noise (0.3) on the slow-changing
# location signals. Framed as an engineering variant, not a paper method.
INPUT_SIGMA_PER_FEATURE = [0.05, 0.1, 0.05, 0.05, 0.05, 0.1, 0.3, 0.3, 0.3]
CLIP_RANGE = 3.0          # clip standardised inputs to [-3, 3] before noise
OUTPUT_SIGMAS = [0.0001, 0.001, 0.005, 0.01]
# DP-SGD is swept over a 2D grid: clip norm × noise multiplier.
# This is the FIXED-CLIP baseline (Abadi et al. 2016); adaptive-clipping papers
# (e.g. Pichapati et al. 2019, DPSGD-Global-Adapt) improve on this.
DPSGD_CLIPS = [0.5, 1.0, 5.0]
DPSGD_NMS   = [1.0, 2.0]        # reduced subset to keep total runs to 6
DELTA = 1e-8              # << 1/(10n) for the ~3.06M-window train split (see METHODS.md)


def _fresh_model():
    return GRUAutoencoder(input_size=len(FEATURES), hidden_size=64, num_layers=2)


def _save_partial_results(results, out_dir):
    """
    Write accumulated result rows to results_table_partial.csv after every completed
    run, so a crash late in the sweep never loses completed runs. Overwritten each
    call; the final formatted results_table.csv is unaffected.
    """
    df = pd.DataFrame(results)
    if "clip_norm" not in df.columns:
        df["clip_norm"] = float("nan")
    if "epsilon_final" not in df.columns:
        df["epsilon_final"] = float("nan")
    df.to_csv(os.path.join(out_dir, "results_table_partial.csv"), index=False)


# Orchestrates the whole experiment: baseline, then the three perturbation families,
# collecting AUROC/F1/ε into one results table and one privacy–utility plot.
def run(data_dir, out_dir, epochs=30, batch_size=256, device=None,
        dpsgd_physical_batch=512):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed BEFORE any model is constructed so sweeps are reproducible across runs
    # (train()/train_dpsgd() only reseed after the model already exists).
    torch.manual_seed(42)

    models_dir = os.path.join(out_dir, "models")
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("Loading sessions …")
    sessions, vehicle_ids = load_sessions_with_vehicle_ids(data_dir)
    (train_s, val_s, test_s), (train_vids, _, _) = split_sessions(
        sessions, auxiliary=vehicle_ids
    )

    # Within-session MI holdout: carve a same-session held-out tail (with a WINDOW-sized
    # guard gap) off each TRAINING session as MI non-members. Members and non-members
    # then share no frames AND come from the same session distribution — cross-session
    # non-members would confound distribution shift with memorisation (see METHODS.md §5).
    train_parts, holdout_parts = split_sessions_for_mi(train_s, holdout_frac=0.1, window=WINDOW)

    train_ds, scaler = make_dataset(train_parts, fit_scaler=True)
    val_ds, _ = make_dataset(val_s, scaler=scaler)
    test_ds, _ = make_dataset(test_s, scaler=scaler)
    mi_nonmember_ds, _ = make_dataset(holdout_parts, scaler=scaler)

    # Session-group labels for the cluster bootstrap CI (resampled at the session
    # level, not per-window — stride-1 windows are heavily autocorrelated).
    test_groups = session_groups_for_windows(test_s, window=WINDOW)
    train_groups = session_groups_for_windows(train_parts, window=WINDOW)
    mi_nonmember_groups = session_groups_for_windows(holdout_parts, window=WINDOW)

    results = []

    def _eval(model):
        """Compute IDS utility metrics + MI privacy metrics for one model.

        MI always uses clean train_ds as members and clean mi_nonmember_ds (same
        sessions, held-out tail) as non-members, regardless of whether the model was
        trained on noisy data. This makes MI-AUC comparable across all methods (noisy
        members would partly measure noise level, not membership).
        """
        m = compute_metrics(model, test_ds, val_dataset=val_ds, device=device, groups=test_groups)

        balanced_members = balance_members(train_ds, len(mi_nonmember_ds), seed=42)
        member_idx = balanced_subsample_indices(len(train_ds), len(mi_nonmember_ds), seed=42)
        balanced_groups = train_groups[member_idx]
        m.update(membership_inference(
            model, balanced_members, mi_nonmember_ds, device=device,
            member_groups=balanced_groups, nonmember_groups=mi_nonmember_groups,
        ))
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
    sensitivity = 2 * CLIP_RANGE  # bounded by clipping → valid local-DP ε
    # Sweep each scalar σ, plus one per-feature σ vector — every entry is its own model.
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
        # For per-feature σ, the worst-case (largest) ε uses the feature with least noise
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

    # ------------------------ output perturbation (last layer only, inspired by Lu et al. 2022)
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
        # DP-SGD 2D sweep: every (clip norm × noise multiplier) pair is a separate run,
        # showing how clipping and noise jointly move the privacy–utility point.
        # Same --epochs as every other family (DP-SGD needs equal-or-more training to
        # reach its achievable utility, not less — De et al. 2022).
        for clip in DPSGD_CLIPS:
            for nm in DPSGD_NMS:
                print(f"  clip={clip}  noise_multiplier={nm}")
                model = _fresh_model()
                model, eps, eps_final = train_dpsgd(
                    model, train_ds, val_ds,
                    save_path=os.path.join(models_dir, f"dpsgd_clip{clip}_nm{nm}.pth"),
                    noise_multiplier=nm, max_grad_norm=clip, delta=DELTA,
                    epochs=epochs, batch_size=batch_size,
                    max_physical_batch_size=dpsgd_physical_batch, device=device,
                )
                metrics = _eval(model)
                results.append({
                    "method": "DP-SGD (fixed-clip baseline)",
                    "sigma": nm, "clip_norm": clip, "epsilon": eps, "epsilon_final": eps_final,
                    **metrics,
                })
                _save_partial_results(results, out_dir)

    # ---------------------------------------- personalized DP (per-vehicle noise)
    print("\n=== Personalized DP (per-vehicle noise) ===")

    # Build per-vehicle training AND holdout datasets from train_parts/holdout_parts
    # (never raw train_s — the model must never see holdout frames, and MI non-members
    # must come from that SAME vehicle's held-out sessions, not a shared cross-vehicle
    # set, which would confound vehicle-level distribution shift with memorisation).
    # We iterate zip(train_parts, train_vids) so no test-set session can appear here.
    _vehicle_tensors = {}
    _vehicle_holdout_tensors = {}
    for df, vid in zip(train_parts, train_vids):
        ds, _ = make_dataset([df], scaler=scaler)
        if len(ds) > 0:
            _vehicle_tensors.setdefault(vid, []).append(ds.tensors[0])
    for df, vid in zip(holdout_parts, train_vids):
        ds, _ = make_dataset([df], scaler=scaler)
        if len(ds) > 0:
            _vehicle_holdout_tensors.setdefault(vid, []).append(ds.tensors[0])

    vehicle_train_ds = {
        vid: TensorDataset(torch.cat(ts, dim=0))
        for vid, ts in _vehicle_tensors.items()
    }
    vehicle_holdout_ds = {
        vid: TensorDataset(torch.cat(ts, dim=0))
        for vid, ts in _vehicle_holdout_tensors.items()
    }

    # Use baseline model to estimate per-vehicle MI vulnerability
    print("  Computing per-vehicle MI AUC from baseline model …")
    # Score each vehicle's leakage (own held-out sessions as non-members) on the clean
    # baseline model → drives its personalised σ.
    v_mi_aucs = per_vehicle_mi(baseline_model, vehicle_train_ds, vehicle_holdout_ds, device=device)
    noise_map = personalized_noise_map(v_mi_aucs)
    print(f"  Vehicle MI AUCs: {v_mi_aucs}")
    print(f"  Noise map:       {noise_map}")

    # Apply each vehicle's own σ, then train ONE global model on the mixed-noise data.
    personalized_train_ds = apply_personalized_input_noise(
        vehicle_train_ds, noise_map, clip_range=CLIP_RANGE
    )
    model = _fresh_model()
    model = train(
        model, personalized_train_ds, val_ds,
        save_path=os.path.join(models_dir, "personalized.pth"),
        epochs=epochs, batch_size=batch_size, device=device,
    )
    # MI uses clean train_ds as members (same as every other method)
    metrics = _eval(model)
    sigma_str = " | ".join(f"v{vid}:{s:.3f}" for vid, s in sorted(noise_map.items()))
    results.append({
        "method": "Personalized", "sigma": sigma_str, "epsilon": float("nan"),
        **metrics,
    })
    _save_partial_results(results, out_dir)

    # ------------------------------------------------------------ table
    df = pd.DataFrame(results)
    # clip_norm / epsilon_final only populated for DP-SGD rows; fill NaN for others
    if "clip_norm" not in df.columns:
        df["clip_norm"] = float("nan")
    if "epsilon_final" not in df.columns:
        df["epsilon_final"] = float("nan")
    df = df[["method", "sigma", "clip_norm", "epsilon", "epsilon_final",
              "auroc", "auroc_lo", "auroc_hi", "f1", "fpr_at_95tpr", "mean_mse_normal",
              "mi_auc", "mi_auc_lo", "mi_auc_hi"]]
    df.columns = ["Method", "σ / noise_mult", "clip_norm", "ε", "ε (final epoch)",
                  "AUROC", "AUROC_lo", "AUROC_hi", "F1", "FPR@95TPR", "MSE(normal)",
                  "MI-AUC", "MI-AUC_lo", "MI-AUC_hi"]
    table_path = os.path.join(out_dir, "results_table.csv")
    df.to_csv(table_path, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nTable saved to {table_path}")

    _plot_epsilon(df, figures_dir, baseline_auroc=df.loc[df["Method"] == "Baseline", "AUROC"].values[0])
    _plot_mi(df, figures_dir, baseline_auroc=df.loc[df["Method"] == "Baseline", "AUROC"].values[0],
             baseline_mi=df.loc[df["Method"] == "Baseline", "MI-AUC"].values[0])

    return df


def _plot_epsilon(df, figures_dir, baseline_auroc):
    """
    Two-panel ε-based plot.
    Left:  Input + Output methods (local / parameter DP).
    Right: DP-SGD (central DP via Opacus RDP accountant).
    Panels are NOT on a shared ε axis — ε is not comparable across them.
    """
    fig, (ax_local, ax_central) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "Privacy–Utility Tradeoff: GRU Autoencoder IDS\n"
        "(ε not comparable across panels — left=local/parameter DP, right=central DP)",
        fontsize=11,
    )

    for ax in (ax_local, ax_central):
        ax.axhline(baseline_auroc, color="black", linestyle="--", linewidth=1,
                   label=f"Baseline AUROC={baseline_auroc:.3f}")
        ax.set_ylabel("IDS AUROC")
        ax.grid(True, which="both", alpha=0.3)

    # ---- left panel: local / parameter DP ----
    local_styles = {
        "Input":              ("steelblue",  "o", "-"),
        "Output":             ("darkorange", "s", "-"),
        "Output (last-layer)":("sienna",     "D", "--"),
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

    # ---- right panel: central DP (DP-SGD) ----
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
    path = os.path.join(figures_dir, "privacy_utility_epsilon.png")
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close(fig)


def _plot_mi(df, figures_dir, baseline_auroc, baseline_mi):
    """
    Single-panel MI-based privacy–utility plot.
    X = MI-AUC (0.5 = perfect privacy, 1.0 = worst).
    Y = IDS AUROC.
    MI-AUC is comparable across all DP families, unlike ε.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(baseline_auroc, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(baseline_mi,    color="black", linestyle=":",  linewidth=1, alpha=0.6,
               label=f"Baseline (AUROC={baseline_auroc:.3f}, MI-AUC={baseline_mi:.3f})")

    style_map = {
        "Input":                       ("steelblue",  "o", "-"),
        "Output":                      ("darkorange", "s", "-"),
        "Output (last-layer)":         ("sienna",     "D", "--"),
        "DP-SGD (fixed-clip baseline)":("seagreen",   "^", "-"),
        "Personalized":                ("purple",     "*", "none"),
    }
    for method, (color, marker, ls) in style_map.items():
        sub = df[df["Method"] == method].copy()
        sub = sub[sub["MI-AUC"].notna()].sort_values("MI-AUC")
        if sub.empty:
            continue
        # Horizontal error bars from the cluster-bootstrap MI-AUC CI (95%, session-level
        # resampling — see evaluate.bootstrap_auc_ci) so small cross-method gaps aren't
        # over-interpreted.
        xerr = np.vstack([
            (sub["MI-AUC"] - sub["MI-AUC_lo"]).clip(lower=0).to_numpy(),
            (sub["MI-AUC_hi"] - sub["MI-AUC"]).clip(lower=0).to_numpy(),
        ])
        fmt = marker if ls == "none" else marker + ls
        ax.errorbar(sub["MI-AUC"], sub["AUROC"], xerr=xerr, fmt=fmt, color=color,
                    label=method, capsize=3, zorder=3)

    ax.set_xlabel("MI Attack AUC  (0.5 = private, 1.0 = fully exposed; error bars = 95% CI)",
                  fontsize=10)
    ax.set_ylabel("IDS AUROC", fontsize=10)
    ax.set_title("Privacy–Utility Tradeoff: MI-AUC as privacy axis\n"
                 "(comparable across local DP, central DP, and personalised DP)")
    # Flip x so 'more private' (MI-AUC → 0.5) sits on the left, like a normal privacy axis.
    ax.invert_xaxis()   # left = more private
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "privacy_utility_mi.png")
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
    parser.add_argument("--data_dir", default="data/CARLA_processed")
    parser.add_argument("--out_dir", default="notebooks")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epoch count, shared by ALL families including DP-SGD "
                             "(patience=5 default in train()/train_dpsgd()).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dpsgd_physical_batch", type=int, default=512,
                        help="Opacus max_physical_batch_size for DP-SGD runs (memory "
                             "only — does not change the logical batch_size or ε).")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, args.epochs, args.batch_size,
        dpsgd_physical_batch=args.dpsgd_physical_batch)
