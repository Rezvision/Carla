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
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.evaluate import (
    balanced_subsample_indices,
    compute_metrics_labeled,
    membership_inference,
    reconstruction_errors,
)
from src.kaggle_data import (
    KAGGLE_FEATURES,
    load_kaggle,
    make_kaggle_dataset,
    split_kaggle_stratified,
)
from src.compare import save_figure  # shared PNG + report-sized vector PDF writer
from src import plotstyle
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
from src.timing import (
    TIMING_COLUMNS, TIMING_HEADERS, run_meta, save_run_timing_meta, time_block,
)

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


def load_kaggle_model(path, device=None):
    """Reconstruct an MLPAutoencoder and load a saved checkpoint (used by src.reeval)."""
    model = _fresh_model()
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
    return model


def prepare_kaggle_data(data_path):
    """
    Load + split the Kaggle telemetry set and build every dataset/label array needed.

    Factored out of run() so src.reeval rebuilds the EXACT same split (same seed/function)
    before scoring saved checkpoints. Returns a dict of datasets, label arrays, the
    normal-only MI non-member set, the fitted scaler, and split-size diagnostics.
    """
    df = load_kaggle(data_path)
    train_df, val_df, test_df = split_kaggle_stratified(df, seed=42)

    train_ds, train_labels, scaler = make_kaggle_dataset(train_df, fit_scaler=True)
    val_ds, val_labels, _ = make_kaggle_dataset(val_df, scaler=scaler)
    test_ds, test_labels, _ = make_kaggle_dataset(test_df, scaler=scaler)

    # MI non-members = only normal test rows (anomalies would bias reconstruction errors).
    test_normal_mask = test_labels == 0
    test_normal_ds = TensorDataset(test_ds.tensors[0][torch.tensor(test_normal_mask)])

    # Join the human-readable failure_type back for figure annotation. load_kaggle keeps
    # only the numeric anomaly_label, but split_kaggle_stratified preserves the ORIGINAL
    # DataFrame index, so the raw CSV can be indexed by test_df.index to recover which
    # kind of failure each test row was (used to label missed detections).
    try:
        raw_types = pd.read_csv(data_path, usecols=["failure_type"])
        test_failure_types = raw_types.loc[test_df.index, "failure_type"].to_numpy()
    except (ValueError, KeyError):
        test_failure_types = np.array([""] * len(test_labels))

    return {
        "train_ds": train_ds, "train_labels": train_labels,
        "val_ds": val_ds, "val_labels": val_labels,
        "test_ds": test_ds, "test_labels": test_labels,
        "test_failure_types": test_failure_types,
        "test_normal_ds": test_normal_ds, "scaler": scaler,
        "n_val_anom": int(val_df["anomaly_label"].sum()),
        "n_test_anom": int(test_df["anomaly_label"].sum()),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    }


def evaluate_model(model, data, device=None):
    """IDS detection metrics (real labels, val threshold, PR-AUC) + MI privacy metrics,
    each with a 95% i.i.d. row-bootstrap CI.

    Shared by run() and src.reeval. Kaggle rows are independent tabular samples (the
    timestamps are shuffled — see kaggle_data.py), so a plain row-level bootstrap is
    valid here, unlike CARLA which needs the session-cluster bootstrap. The CIs are wide
    (~18 real test positives) and that width IS the finding: it is what tells the reader
    the cross-method gaps are noise.
    """
    m = compute_metrics_labeled(model, data["test_ds"], data["test_labels"],
                                val_dataset=data["val_ds"], val_labels=data["val_labels"],
                                device=device, iid_ci=True)
    m.update(membership_inference(model, data["train_ds"], data["test_normal_ds"],
                                  device=device, iid_ci=True))
    return m


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
    data = prepare_kaggle_data(data_path)
    train_ds = data["train_ds"]
    val_ds = data["val_ds"]
    n_test_anom = data["n_test_anom"]
    print(f"  Train: {data['n_train']} normal rows  (0 anomalies by design)")
    print(f"  Val:   {data['n_val']} rows, {data['n_val_anom']} anomalies")
    print(f"  Test:  {data['n_test']} rows, {n_test_anom} anomalies")
    if n_test_anom < 5:
        print(f"  WARNING: only {n_test_anom} anomalies in test set — AUROC will be high-variance")
    # PR-AUC on ~18 real test positives is high-variance. Kaggle rows are independent
    # samples, so an i.i.d. row bootstrap quantifies that variance — expect wide CIs.
    print(f"  NOTE: PR-AUC is computed on {n_test_anom} test positives with a 95% i.i.d. "
          f"row-bootstrap CI — expect wide intervals; treat overlapping gaps as noise.")

    results = []
    meta = run_meta(device)  # device + hostname, identical on every row of this run

    # Timed once per run (TIMING_TASK.md part B): bootstrap CIs dominate evaluation.
    eval_seconds_per_model = float("nan")

    def _eval(model):
        nonlocal eval_seconds_per_model
        if math.isnan(eval_seconds_per_model):
            with time_block(sync_device=device) as t:
                out = evaluate_model(model, data, device=device)
            eval_seconds_per_model = t.seconds
            print(f"  [timing] one full evaluate_model() = {t.seconds:.1f}s "
                  f"(bootstrap CIs dominate; measured once per run)")
            return out
        return evaluate_model(model, data, device=device)

    # ------------------------------------------------------------ baseline
    print("\n=== Baseline ===")
    baseline_model = _fresh_model()
    baseline_model, timing = train(
        baseline_model, train_ds, val_ds,
        save_path=os.path.join(models_dir, "baseline.pth"),
        epochs=epochs, batch_size=batch_size, device=device,
    )
    metrics = _eval(baseline_model)
    results.append({"method": "Baseline", "sigma": None, "epsilon": float("inf"),
                    **metrics, **timing.as_row(), **meta})
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
        model, timing = train(
            model, noisy_train_ds, val_ds,
            save_path=os.path.join(models_dir, f"input_sigma{label}.pth"),
            epochs=epochs, batch_size=batch_size, device=device,
        )
        eps = analytic_gaussian_epsilon(worst_case_sigma(sigma), sensitivity=sensitivity, delta=DELTA)
        metrics = _eval(model)
        results.append({"method": "Input", "sigma": label, "epsilon": eps,
                        **metrics, **timing.as_row(), **meta})
        _save_partial_results(results, out_dir)

    # -------------------------------------------------- output perturbation (full model)
    print("\n=== Output Perturbation (full model) ===")
    for sigma in OUTPUT_SIGMAS:
        print(f"  σ={sigma}")
        # Derived, not trained: record the ~0 cost explicitly rather than leaving it blank.
        with time_block(sync_device=device) as t:
            noisy_model = apply_output_noise(baseline_model, sigma=sigma)
        eps = output_noise_epsilon(baseline_model, sigma=sigma, delta=DELTA)
        metrics = _eval(noisy_model)
        results.append({"method": "Output", "sigma": sigma, "epsilon": eps,
                        **metrics, "derive_seconds": t.seconds, **meta})
        _save_partial_results(results, out_dir)

    # ----------------------------------- output perturbation (last layer only)
    print("\n=== Output Perturbation (last-layer only) ===")
    for sigma in OUTPUT_SIGMAS:
        print(f"  σ={sigma}")
        with time_block(sync_device=device) as t:
            noisy_model = apply_output_noise_last_layer(baseline_model, sigma=sigma)
        eps = output_last_layer_epsilon(baseline_model, sigma=sigma, delta=DELTA)
        metrics = _eval(noisy_model)
        results.append({"method": "Output (last-layer)", "sigma": sigma, "epsilon": eps,
                        **metrics, "derive_seconds": t.seconds, **meta})
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
                model, eps, eps_final, timing = train_dpsgd(
                    model, train_ds, val_ds,
                    save_path=os.path.join(models_dir, f"dpsgd_clip{clip}_nm{nm}.pth"),
                    noise_multiplier=nm, max_grad_norm=clip, delta=DELTA,
                    epochs=epochs, batch_size=batch_size, device=device,
                )
                metrics = _eval(model)
                results.append({
                    "method": "DP-SGD (fixed-clip baseline)",
                    "sigma": nm, "clip_norm": clip, "epsilon": eps, "epsilon_final": eps_final,
                    **metrics, **timing.as_row(), **meta,
                })
                _save_partial_results(results, out_dir)

    # ------------------------------------------------------------ table + figures
    df_out = build_results_table(results, out_dir)
    print(f"\n{df_out.to_string(index=False)}")
    print(f"\nTable saved to {os.path.join(out_dir, 'kaggle_results_table.csv')}")
    save_run_timing_meta(out_dir, eval_seconds_per_model, device,
                         name="kaggle_timing_run_meta.csv")

    make_figures(df_out, figures_dir, models_dir, data, device, baseline_model=baseline_model)

    return df_out


# Kaggle table schema: PR-AUC (headline utility, real labels) + eval prevalence, then
# the existing AUROC/F1/FPR/MSE utility columns and the MI-AUC privacy column. Each
# ranking metric carries a 95% i.i.d. row-bootstrap CI (valid here because Kaggle rows
# are independent samples; CARLA's table uses the cluster bootstrap instead).
_KAGGLE_COLUMNS = [
    "method", "sigma", "clip_norm", "epsilon", "epsilon_final",
    "pr_auc", "pr_auc_lo", "pr_auc_hi", "prevalence",
    "auroc", "auroc_lo", "auroc_hi",
    "f1", "fpr_at_95tpr", "mean_mse_normal",
    "mi_auc", "mi_auc_lo", "mi_auc_hi",
    # Cost axis (TIMING_TASK.md part B) — same columns as the CARLA table.
    *TIMING_COLUMNS,
]
_KAGGLE_HEADERS = [
    "Method", "σ / noise_mult", "clip_norm", "ε", "ε (final epoch)",
    "PR-AUC", "PR-AUC lo", "PR-AUC hi", "prevalence",
    "AUROC", "AUROC lo", "AUROC hi",
    "F1", "FPR@95TPR", "MSE(normal)",
    "MI-AUC", "MI-AUC lo", "MI-AUC hi",
    *TIMING_HEADERS,
]


def build_results_table(results, out_dir):
    """Format accumulated Kaggle result rows into kaggle_results_table.csv (shared with reeval)."""
    df = pd.DataFrame(results)
    for col in _KAGGLE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[_KAGGLE_COLUMNS]
    df.columns = _KAGGLE_HEADERS
    table_path = os.path.join(out_dir, "kaggle_results_table.csv")
    df.to_csv(table_path, index=False)
    return df


def make_figures(df, figures_dir, models_dir, data, device, baseline_model=None):
    """
    Regenerate all Kaggle figures from a results table (shared with reeval).

    kaggle_score_strip.png is the headline (raw scores, nothing aggregated);
    kaggle_dotplot.png is the report summary (with CIs); kaggle_pr_curves.png is appendix
    material.
    """
    _plot_score_strip(models_dir, data, device, figures_dir, baseline_model=baseline_model)
    _plot_dotplot(df, figures_dir)
    _plot_pr_curves(models_dir, data, device, figures_dir, baseline_model=baseline_model)
    _remove_retired_figures(figures_dir)


# Kaggle figures whose ROLE was taken over by another figure. Deleted on every run so a
# stale copy can never be picked up for the deck by mistake — same contract as
# compare._RETIRED_FIGURES.
_RETIRED_FIGURES = {
    # The whisker-free "slides" dotplot existed so a slide could show the dots without the
    # CIs. But the CIs ARE the Kaggle finding — every interval overlaps, so nothing here is
    # distinguishable — and a slide that drops them keeps the ordering while discarding the
    # only thing that says the ordering means nothing. kaggle_score_strip.png is the slide;
    # this figure's job was already covered and its version of it was misleading.
    "kaggle_dotplot_slides.png": "score_strip is the slide; the CI dotplot covers the report",
}


def _remove_retired_figures(figures_dir):
    pdf_dir = os.path.join(os.path.dirname(figures_dir.rstrip(os.sep)) or ".", "figures_pdf")
    for name, why in _RETIRED_FIGURES.items():
        for path in (os.path.join(figures_dir, name),
                     os.path.join(pdf_dir, os.path.splitext(name)[0] + ".pdf")):
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed retired {path} ({why})")


# The Kaggle figures share the CARLA palette (src.plotstyle) so both datasets read as one
# family of plots. Colour only — the per-family marker shapes this dict used to carry are
# retired study-wide (VIZ_REDESIGN_TASK.md rule 2). Identity is not colour-alone even so:
# every row of the dot plot is named on the y-axis in its family's colour.
_FAMILY_ORDER = ["Baseline", "Input", "Output", "Output (last-layer)",
                 plotstyle.DPSGD_METHOD]

# Spec limits (KAGGLE_FIGURE_TASK.md): keep MI near 0.5 so proximity to chance is honest
# rather than magnified, and keep PR-AUC far from a misleading zoom. The axes only ever
# EXPAND from these to fit a CI whisker — never shrink — so no interval is truncated.
_PR_XLIM = (0.90, 0.96)
_MI_XLIM = (0.49, 0.56)


def _sigma_sort_key(v):
    """Order σ cells: numeric ascending, non-numeric labels ('per-feature') last."""
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, 0.0)


def _fmt_sigma(v):
    """Display a σ / clip cell: floats and numeric strings print alike ('0.01', '0.5')."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _row_label(method, row):
    sigma = _fmt_sigma(row["σ / noise_mult"])
    if method == "Baseline":
        return "Baseline"
    if method == "Input":
        return f"Input  σ={sigma}"
    if method == "Output":
        return f"Output  σ={sigma}"
    if method == "Output (last-layer)":
        return f"Output-LL  σ={sigma}"
    # DP-SGD keeps ε in its label: this figure replaced the ε plot, so it is the only
    # place a reviewer sees ε without opening the results table.
    eps = row.get("ε", float("nan"))
    eps_txt = ""
    if pd.notna(eps) and np.isfinite(float(eps)):
        eps_txt = f", ε={float(eps):.1f}"
    return f"DP-SGD  C={_fmt_sigma(row['clip_norm'])}, nm={sigma}{eps_txt}"


def _dotplot_rows(df):
    """
    One entry per condition in display order (top→bottom: Baseline, Input, Output,
    Output-LL, DP-SGD), plus the row indices where a new family starts (separator lines).
    """
    rows, boundaries = [], []
    for method in _FAMILY_ORDER:
        sub = [r for _, r in df[df["Method"] == method].iterrows()]
        if not sub:
            continue
        if method == "DP-SGD (fixed-clip baseline)":
            sub.sort(key=lambda r: (_sigma_sort_key(r["clip_norm"]),
                                    _sigma_sort_key(r["σ / noise_mult"])))
        else:
            sub.sort(key=lambda r: _sigma_sort_key(r["σ / noise_mult"]))
        if rows:
            boundaries.append(len(rows))
        for r in sub:
            rows.append({"method": method, "label": _row_label(method, r), "row": r})
    return rows, boundaries


def _draw_facet(ax, rows, value_col, lo_col, hi_col, xlim):
    """Dots + horizontal bootstrap-CI error bars for one metric; returns the x-range
    actually needed (spec limits, expanded to fit any whisker outside)."""
    lo_lim, hi_lim = xlim
    for i, item in enumerate(rows):
        color = plotstyle.family_color(item["method"])
        r = item["row"]
        val = float(r[value_col])
        lo, hi = r.get(lo_col, float("nan")), r.get(hi_col, float("nan"))

        xerr = None
        if pd.notna(lo) and pd.notna(hi):
            lo, hi = float(lo), float(hi)
            # A percentile CI can land entirely on one side of the point estimate; clamp
            # at 0 so matplotlib never receives a negative error-bar length.
            xerr = [[max(val - lo, 0.0)], [max(hi - val, 0.0)]]
            lo_lim, hi_lim = min(lo_lim, lo), max(hi_lim, hi)
        lo_lim, hi_lim = min(lo_lim, val), max(hi_lim, val)

        ax.errorbar(val, i, xerr=xerr, fmt=plotstyle.MARKER, color=color, markersize=7,
                    elinewidth=1.0, capsize=0, ecolor=to_rgba(color, 0.45), zorder=3)
    return lo_lim, hi_lim


def _plot_dotplot(df, figures_dir):
    """
    Cleveland dot plot: one row per condition, two facets sharing the y-axis
    (left = PR-AUC utility, right = MI-AUC privacy).

    Why a dot plot and not the old line/scatter figures: the Kaggle conditions are ~20
    discrete categories whose metrics are saturated and trendless, so lines implied an
    ordering and a tradeoff that do not exist on this dataset. Why not bars: these values
    cluster in a narrow band far from 0, and a bar encodes value BY LENGTH, so it needs a
    zero baseline — at which every bar looks identical, while truncating the axis to make
    them differ actively misleads. A dot encodes value by position and carries no
    zero-baseline requirement, so a truncated axis is legitimate.

    ALWAYS with whiskers. There used to be a second, whisker-free copy of this figure for
    slides; it is retired (see _RETIRED_FIGURES) because on this dataset the overlapping
    CIs are the entire finding, and a version that keeps the row ordering while dropping
    the intervals shows a ranking that the intervals exist to deny.

    Report appendix: kaggle_score_strip.png is the slide.
    """
    rows, boundaries = _dotplot_rows(df)
    if not rows:
        print("  skip dot plot (no rows in results table)")
        return

    baseline = df[df["Method"] == "Baseline"]
    baseline_pr = float(baseline["PR-AUC"].values[0]) if not baseline.empty else None
    prevalence = float(df["prevalence"].dropna().iloc[0]) if df["prevalence"].notna().any() else None

    fig, (ax_pr, ax_mi) = plt.subplots(1, 2, figsize=(13.5, 8.0), sharey=True)

    pr_lim = _draw_facet(ax_pr, rows, "PR-AUC", "PR-AUC lo", "PR-AUC hi", _PR_XLIM)
    mi_lim = _draw_facet(ax_mi, rows, "MI-AUC", "MI-AUC lo", "MI-AUC hi", _MI_XLIM)

    # Reference lines are direct-labelled above the plot rather than given a legend box:
    # every row already runs an error bar across the panel, so a box would sit on data.
    # One lowercase word each, matching every other figure in the study — the value and the
    # "(perfect privacy)" gloss these used to carry are in the doc.
    if baseline_pr is not None:
        ax_pr.axvline(baseline_pr, color=plotstyle.BASELINE_COLOR, linestyle="--",
                      linewidth=1, alpha=0.7, zorder=1)
        ax_pr.text(baseline_pr, -1.05, plotstyle.BASELINE_LABEL, ha="center",
                   va="bottom", fontsize=8, color=plotstyle.MUTED, zorder=4,
                   bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax_mi.axvline(0.5, color=plotstyle.AXIS, linewidth=1, alpha=0.9, zorder=1)
    ax_mi.text(0.5, -1.05, plotstyle.CHANCE_LABEL, ha="center", va="bottom",
               fontsize=8, color=plotstyle.MUTED, zorder=4,
               bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    for ax, lim in ((ax_pr, pr_lim), (ax_mi, mi_lim)):
        for b in boundaries:
            ax.axhline(b - 0.5, color="gray", linewidth=0.6, alpha=0.45, zorder=0)
        pad = 0.02 * (lim[1] - lim[0])
        ax.set_xlim(lim[0] - pad, lim[1] + pad)
        ax.grid(False, axis="y")
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)

    ax_pr.set_yticks(np.arange(len(rows)))
    ax_pr.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax_pr.set_ylim(len(rows) - 0.5, -1.5)  # first row at the top; headroom for the ref labels
    for tick, item in zip(ax_pr.get_yticklabels(), rows):
        tick.set_color(plotstyle.family_color(item["method"]))

    pr_note = f"; no-skill = prevalence ≈ {prevalence:.3f}" if prevalence is not None else ""
    ax_pr.set_xlabel(f"PR-AUC — utility (real failure labels{pr_note})", fontsize=10)
    ax_mi.set_xlabel("MI attack AUC — privacy (0.5 = chance, higher = more exposed)", fontsize=10)
    ax_pr.set_title("Utility", fontsize=10.5)
    ax_mi.set_title("Privacy", fontsize=10.5)

    # That every utility CI overlaps the baseline's and every MI-AUC CI straddles chance \u2014
    # so the row ordering carries no finding at all \u2014 is the figure's paragraph in the doc.
    title = "Utility and privacy by configuration (Kaggle)"

    def relayout(f):
        plotstyle.style_figure(f, title)

    relayout(fig)
    save_figure(fig, figures_dir, "kaggle_dotplot.png", relayout=relayout)
    plt.close(fig)


def _representative_models(models_dir, device, baseline_model=None):
    """
    (label, model, colour) triples shared by the PR-curve and score-strip figures:
    baseline + one mid-noise model per family, so the two figures always describe the
    SAME set of models. Output is derived from the baseline via seeded noise (it is never
    checkpointed). No personalized family on Kaggle. Missing checkpoints are skipped.
    Colours come from src.plotstyle so every figure in the study agrees on what each family
    looks like.
    """
    if baseline_model is None:
        baseline_model = load_kaggle_model(os.path.join(models_dir, "baseline.pth"), device)

    reps = [("Baseline", baseline_model, plotstyle.BASELINE_COLOR)]

    input_path = os.path.join(models_dir, "input_sigma0.1.pth")
    if os.path.exists(input_path):
        reps.append(("Input σ=0.1", load_kaggle_model(input_path, device),
                     plotstyle.family_color("Input")))

    reps.append(("Output σ=0.005", apply_output_noise(baseline_model, sigma=0.005),
                 plotstyle.family_color("Output")))

    dpsgd_path = os.path.join(models_dir, "dpsgd_clip1.0_nm1.0.pth")
    if os.path.exists(dpsgd_path):
        reps.append(("DP-SGD C=1, nm=1", load_kaggle_model(dpsgd_path, device),
                     plotstyle.family_color(plotstyle.DPSGD_METHOD)))
    return reps


def _plot_score_strip(models_dir, data, device, figures_dir, baseline_model=None,
                      max_normals=1000, seed=42):
    """
    Raw anomaly scores per representative model — the headline Kaggle figure.

    With only ~18 positives, every aggregate curve on this dataset is a quantisation
    staircase, so this figure aggregates NOTHING: each red point is one real failure and
    each grey point one normal test row, exactly as the model scored them. There is no
    estimator here, so there is nothing to put an error bar on — the spread the reader
    sees IS the data. Every column scores the same test rows, so differences between
    columns are differences between models, not between samples.
    """
    reps = _representative_models(models_dir, device, baseline_model=baseline_model)
    labels = np.asarray(data["test_labels"])
    is_fail = labels == 1
    n_fail = int(is_fail.sum())

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    downsampled = False

    for i, (label, model, color) in enumerate(reps):
        errors = reconstruction_errors(model, data["test_ds"], device=device)
        normal_errors = errors[~is_fail]
        fail_errors = errors[is_fail]

        # Seeded downsample so a dense normal cloud can't become an unreadable block.
        idx = balanced_subsample_indices(len(normal_errors), max_normals, seed=seed)
        if len(idx) < len(normal_errors):
            downsampled = True
        normal_errors = normal_errors[idx]

        # Jitter normals only — 18 failures at one x never overplot, and keeping them on
        # a true vertical line makes their position exactly comparable across columns.
        jitter = rng.uniform(-0.18, 0.18, size=len(normal_errors))
        ax.scatter(i + jitter, normal_errors, s=8, color="gray", alpha=0.25, linewidths=0,
                   zorder=2, label="normal test rows" if i == 0 else None)
        ax.scatter(np.full(len(fail_errors), i, dtype=float), fail_errors, s=34,
                   color="crimson", alpha=0.85, edgecolors="white", linewidths=0.5,
                   zorder=3, label="real failures" if i == 0 else None)

        threshold = compute_metrics_labeled(
            model, data["test_ds"], labels, val_dataset=data["val_ds"],
            val_labels=data["val_labels"], device=device,
        )["threshold"]
        ax.hlines(threshold, i - 0.32, i + 0.32, color="#333333", linestyle="--",
                  linewidth=1.2, zorder=4,
                  label="decision threshold (Youden J on val)" if i == 0 else None)

        # Name the failures the model MISSES (below its threshold): these few points are
        # what cap PR-AUC, so which kind of failure they are is the actionable detail.
        fail_types = np.asarray(data.get("test_failure_types", [""] * len(labels)))[is_fail]
        missed = np.where(fail_errors < threshold)[0]
        for rank, j in enumerate(missed):
            ax.annotate(str(fail_types[j]), (i, fail_errors[j]),
                        textcoords="offset points", xytext=(9, -3 + 9 * (rank % 2)),
                        fontsize=6.5, color="crimson", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    ax.set_yscale("log")
    # Minor log ticks: with everything inside ~1 decade, decade-only ticks make the
    # vertical spread hard to read off.
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                                  numticks=100))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="y", which="minor", length=2.5)
    ax.grid(True, axis="y", which="minor", alpha=0.15)
    ax.set_xticks(np.arange(len(reps)))
    ax.set_xticklabels([r[0] for r in reps], fontsize=9)
    for tick, (_, _, color) in zip(ax.get_xticklabels(), reps):
        tick.set_color(color)  # family colours, matching kaggle_dotplot.png
    ax.set_xlim(-0.5, len(reps) - 0.5)
    ax.set_ylabel("Reconstruction MSE (log scale)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # Below the axes: inside, the legend box covers real failure points in column 1.
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3,
              frameon=False)

    # THE Kaggle slide. That every column looks alike, that the one or two failures inside
    # the normal cloud are what cap PR-AUC, and that normals are seeded-downsampled for
    # overplotting \u2014 all of it is the figure's section in the doc. The legend, the failure
    # markers, the per-failure type annotations and the threshold stay: they are labels on
    # the marks, which is exactly what a stripped figure is allowed to keep.
    title = "Anomaly scores per model (Kaggle)"

    def relayout(f):
        plotstyle.style_figure(f, title)

    relayout(fig)
    save_figure(fig, figures_dir, "kaggle_score_strip.png", relayout=relayout)
    plt.close(fig)


def _plot_pr_curves(models_dir, data, device, figures_dir, baseline_model=None):
    """
    Full precision–recall curves (REAL labels) for representative models. Horizontal
    dashed line at prevalence = the no-skill baseline. Each curve labelled with its
    PR-AUC. High-variance (~18 test positives) — see the printed warning in run().

    Report/appendix figure, not the headline: with 18 positives these are quantisation
    staircases, so kaggle_score_strip.png (which aggregates nothing) carries the story.
    Curves are drawn step-post — precision is only defined AT each threshold, and linear
    interpolation between those points draws precision the classifier never achieves.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    reps = _representative_models(models_dir, device, baseline_model=baseline_model)

    labels = data["test_labels"]
    prevalence = float(np.mean(labels))
    for label, model, color in reps:
        errors = reconstruction_errors(model, data["test_ds"], device=device)
        precision, recall, _ = precision_recall_curve(labels, errors)
        ap = average_precision_score(labels, errors)
        # These curves overlap almost exactly; thin + semi-transparent keeps each visible.
        ax.plot(recall, precision, color=color, linewidth=1.2, alpha=0.75,
                drawstyle="steps-post", label=f"{label}  (PR-AUC={ap:.3f})")

    ax.axhline(prevalence, color=plotstyle.AXIS, linestyle="--", linewidth=1.0,
               label=f"no skill ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    # Centre-left, not the usual upper-right. With ~18 positives every curve holds
    # precision 1.0 until recall ~0.8, so the whole top edge is occupied and a frameless
    # legend there prints its labels straight through four overlapping staircases. The
    # band below them is the one large empty region on this figure.
    plotstyle.small_legend(ax, loc="center left")

    # ONE overlaid panel, deliberately not faceted: that the curves land on top of each
    # other IS the finding, and facetting would hand each model its own panel and destroy
    # the only comparison the figure makes. That they are quantisation staircases at ~18
    # positives, and that the gaps between them are noise, is in the doc.
    title = "Precision\u2013recall on real failure labels"

    def relayout(f):
        plotstyle.style_figure(f, title)

    relayout(fig)
    save_figure(fig, figures_dir, "kaggle_pr_curves.png", relayout=relayout)
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
