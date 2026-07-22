"""
Run baseline + all three DP perturbation families and collect results.

Usage (from repo root):
    python -m src.compare --data_dir data/CARLA_processed --out_dir notebooks
"""
import argparse
import copy
import math
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import torch
from torch.utils.data import TensorDataset
from sklearn.metrics import precision_recall_curve, average_precision_score

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
from src.evaluate import (
    compute_metrics, membership_inference, balance_members, balanced_subsample_indices,
    _detection_eval_set, ATTACK_TYPE_NAMES,
)
from src.anchors import (
    ATTACK_AXIS_LABEL, EXCLUSION_NOTE, FIGURE_EXCLUDED, MI_XLIM, SHORT_NAME,
    load_operating_points, resolve_anchors,
)
from src import plotstyle
from src.personalized import personalized_noise_map, apply_personalized_input_noise, per_vehicle_mi
from src.timing import (
    TIMING_COLUMNS, TIMING_HEADERS, TrainTiming, run_meta, save_run_timing_meta, time_block,
)


# Detection utility is now reported as PR-AUC on a realistic-prevalence eval set (not
# AUROC on a balanced one — precision, and therefore PR-AUC, depends on prevalence).
# 5% is the default synthetic attack prevalence; expose it as a CLI arg.
DEFAULT_ATTACK_PREVALENCE = 0.05


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


def load_carla_model(path, device=None):
    """Reconstruct a GRUAutoencoder and load a saved checkpoint into it.

    Used by src.reeval to re-evaluate saved models without retraining. The architecture
    must match what run() constructed via _fresh_model().

    DP-SGD checkpoints need one extra step: train_dpsgd runs ModuleValidator.fix, which
    swaps each nn.GRU for an Opacus DPGRU. A DPGRU's state_dict carries the standard flat
    GRU parameters (encoder.weight_ih_l0, …) AND per-cell duplicates that share storage
    with them (encoder.l0.ih.weight, …), so a strict load into a plain GRUAutoencoder
    fails on the 16 extra keys even though every parameter it needs is present and
    correct. Drop the DPGRU-only extras and load the flat set — but still fail loudly if
    a parameter the model actually needs is missing, so a genuinely wrong checkpoint
    can't load silently.
    """
    model = _fresh_model()
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    expected = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in expected}
    missing = set(expected) - set(filtered)
    if missing:
        raise RuntimeError(f"checkpoint {path} is missing parameters: {sorted(missing)}")
    model.load_state_dict(filtered)  # strict: shapes must match
    if device is not None:
        model.to(device)
    return model


def prepare_carla_data(data_dir):
    """
    Load CARLA sessions and build every dataset/group array the sweep and re-eval need.

    Factored out of run() so src.reeval rebuilds the EXACT same splits (same seeds and
    functions — no copy-pasting) before scoring saved checkpoints. Returns a dict of:
      train_ds, val_ds, test_ds, mi_nonmember_ds   — the model datasets
      test_groups, train_groups, mi_nonmember_groups — session-level bootstrap groups
      scaler, train_parts, holdout_parts, train_vids — for the personalized section
    """
    sessions, vehicle_ids = load_sessions_with_vehicle_ids(data_dir)
    (train_s, val_s, test_s), (train_vids, _, _) = split_sessions(
        sessions, auxiliary=vehicle_ids
    )

    # Within-session MI holdout (see run() / METHODS.md §5 for the rationale).
    train_parts, holdout_parts = split_sessions_for_mi(train_s, holdout_frac=0.1, window=WINDOW)

    train_ds, scaler = make_dataset(train_parts, fit_scaler=True)
    val_ds, _ = make_dataset(val_s, scaler=scaler)
    test_ds, _ = make_dataset(test_s, scaler=scaler)
    mi_nonmember_ds, _ = make_dataset(holdout_parts, scaler=scaler)

    # Session-group labels for the cluster bootstrap CI (resampled at the session level).
    test_groups = session_groups_for_windows(test_s, window=WINDOW)
    train_groups = session_groups_for_windows(train_parts, window=WINDOW)
    mi_nonmember_groups = session_groups_for_windows(holdout_parts, window=WINDOW)

    return {
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "mi_nonmember_ds": mi_nonmember_ds,
        "test_groups": test_groups, "train_groups": train_groups,
        "mi_nonmember_groups": mi_nonmember_groups,
        "scaler": scaler, "train_parts": train_parts,
        "holdout_parts": holdout_parts, "train_vids": train_vids,
    }


def evaluate_model(model, data, device=None, attack_prevalence=DEFAULT_ATTACK_PREVALENCE):
    """Compute IDS utility metrics + MI privacy metrics for one model.

    Shared by run() and src.reeval so both score models identically. Utility metrics are
    computed on a prevalence-controlled eval set (attack_prevalence) so PR-AUC is
    meaningful; AUROC is reported alongside it (prevalence-invariant).

    MI always uses clean train_ds as members and clean mi_nonmember_ds (same sessions,
    held-out tail) as non-members, regardless of whether the model trained on noisy
    data — this keeps MI-AUC comparable across all methods.
    """
    m = compute_metrics(
        model, data["test_ds"], val_dataset=data["val_ds"], device=device,
        groups=data["test_groups"], attack_prevalence=attack_prevalence,
    )

    balanced_members = balance_members(data["train_ds"], len(data["mi_nonmember_ds"]), seed=42)
    member_idx = balanced_subsample_indices(len(data["train_ds"]), len(data["mi_nonmember_ds"]), seed=42)
    balanced_groups = data["train_groups"][member_idx]
    m.update(membership_inference(
        model, balanced_members, data["mi_nonmember_ds"], device=device,
        member_groups=balanced_groups, nonmember_groups=data["mi_nonmember_groups"],
    ))
    return m


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
        dpsgd_physical_batch=512, attack_prevalence=DEFAULT_ATTACK_PREVALENCE):
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
    data = prepare_carla_data(data_dir)
    # Unpack the pieces the training + personalized sections still need by name.
    train_ds = data["train_ds"]
    val_ds = data["val_ds"]
    scaler = data["scaler"]
    train_parts = data["train_parts"]
    holdout_parts = data["holdout_parts"]
    train_vids = data["train_vids"]

    results = []
    meta = run_meta(device)  # device + hostname, identical on every row of this run

    # Timed once per run (TIMING_TASK.md part B): the bootstrap CIs dominate evaluation,
    # so readers ask how much of the wall clock is eval rather than training. Measuring it
    # on every model would just re-measure the same thing ~40 times.
    eval_seconds_per_model = float("nan")

    def _eval(model):
        nonlocal eval_seconds_per_model
        if math.isnan(eval_seconds_per_model):
            with time_block(sync_device=device) as t:
                out = evaluate_model(model, data, device=device,
                                     attack_prevalence=attack_prevalence)
            eval_seconds_per_model = t.seconds
            print(f"  [timing] one full evaluate_model() = {t.seconds:.1f}s "
                  f"(bootstrap CIs dominate; measured once per run)")
            return out
        return evaluate_model(model, data, device=device, attack_prevalence=attack_prevalence)

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
    sensitivity = 2 * CLIP_RANGE  # bounded by clipping → valid local-DP ε
    # Sweep each scalar σ, plus one per-feature σ vector — every entry is its own model.
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
        # For per-feature σ, the worst-case (largest) ε uses the feature with least noise
        eps = analytic_gaussian_epsilon(worst_case_sigma(sigma), sensitivity=sensitivity, delta=DELTA)
        metrics = _eval(model)
        results.append({"method": "Input", "sigma": label, "epsilon": eps,
                        **metrics, **timing.as_row(), **meta})
        _save_partial_results(results, out_dir)

    # -------------------------------------------------- output perturbation (full model)
    print("\n=== Output Perturbation (full model) ===")
    for sigma in OUTPUT_SIGMAS:
        print(f"  σ={sigma}")
        # Derived, not trained: time the noise application so the ~0 cost is recorded
        # rather than left blank (TIMING_TASK.md part B).
        with time_block(sync_device=device) as t:
            noisy_model = apply_output_noise(baseline_model, sigma=sigma)
        eps = output_noise_epsilon(baseline_model, sigma=sigma, delta=DELTA)
        metrics = _eval(noisy_model)
        results.append({"method": "Output", "sigma": sigma, "epsilon": eps,
                        **metrics, "derive_seconds": t.seconds, **meta})
        _save_partial_results(results, out_dir)

    # ------------------------ output perturbation (last layer only, inspired by Lu et al. 2022)
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
        # DP-SGD 2D sweep: every (clip norm × noise multiplier) pair is a separate run,
        # showing how clipping and noise jointly move the privacy–utility point.
        # Same --epochs as every other family (DP-SGD needs equal-or-more training to
        # reach its achievable utility, not less — De et al. 2022).
        for clip in DPSGD_CLIPS:
            for nm in DPSGD_NMS:
                print(f"  clip={clip}  noise_multiplier={nm}")
                model = _fresh_model()
                model, eps, eps_final, timing = train_dpsgd(
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
                    **metrics, **timing.as_row(), **meta,
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
    model, timing = train(
        model, personalized_train_ds, val_ds,
        save_path=os.path.join(models_dir, "personalized.pth"),
        epochs=epochs, batch_size=batch_size, device=device,
    )
    # MI uses clean train_ds as members (same as every other method)
    metrics = _eval(model)
    sigma_str = " | ".join(f"v{vid}:{s:.3f}" for vid, s in sorted(noise_map.items()))
    results.append({
        "method": "Personalized", "sigma": sigma_str, "epsilon": float("nan"),
        **metrics, **timing.as_row(), **meta,
    })
    _save_partial_results(results, out_dir)

    # ------------------------------------------------------------ table + figures
    df = build_results_table(results, out_dir)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nTable saved to {os.path.join(out_dir, 'results_table.csv')}")
    save_run_timing_meta(out_dir, eval_seconds_per_model, device)

    make_figures(df, figures_dir, models_dir, data, device, attack_prevalence,
                 baseline_model=baseline_model)

    return df


# Final table schema: PR-AUC (headline utility) + its CI + eval prevalence, then the
# existing AUROC/F1/FPR/MSE utility columns and the MI-AUC privacy columns.
_RESULT_COLUMNS = [
    "method", "sigma", "clip_norm", "epsilon", "epsilon_final",
    "pr_auc", "pr_auc_lo", "pr_auc_hi", "prevalence",
    "auroc", "auroc_lo", "auroc_hi", "f1", "fpr_at_95tpr", "mean_mse_normal",
    # Per-attack-type detection (explains the pr_curves recall ceiling): fuzzy = easy,
    # plateau = hard.
    "recall_fuzzy", "recall_plateau", "auroc_fuzzy", "auroc_plateau",
    # Privacy: raw MI-AUC AND effective MI-AUC = max(raw, 1-raw). A raw value < 0.5 is an
    # inverted (still-leaking) attack, not extra privacy — effective is the honest axis.
    "mi_auc", "mi_auc_lo", "mi_auc_hi",
    "mi_auc_effective", "mi_auc_eff_lo", "mi_auc_eff_hi",
    # Cost (TIMING_TASK.md part B): the third axis. train_seconds/epochs_ran/sec_per_epoch
    # are populated for TRAINED models; derive_seconds for models derived from an existing
    # checkpoint by seeded noise (output families) — ~0, which is itself the finding.
    *TIMING_COLUMNS,
]
_RESULT_HEADERS = [
    "Method", "σ / noise_mult", "clip_norm", "ε", "ε (final epoch)",
    "PR-AUC", "PR-AUC_lo", "PR-AUC_hi", "prevalence",
    "AUROC", "AUROC_lo", "AUROC_hi", "F1", "FPR@95TPR", "MSE(normal)",
    "recall_fuzzy", "recall_plateau", "AUROC_fuzzy", "AUROC_plateau",
    "MI-AUC", "MI-AUC_lo", "MI-AUC_hi",
    "MI-AUC_eff", "MI-AUC_eff_lo", "MI-AUC_eff_hi",
    *TIMING_HEADERS,
]


def build_results_table(results, out_dir):
    """Format the accumulated result rows into results_table.csv (shared with reeval).

    PR-AUC is the headline utility metric (finite only under the prevalence-controlled
    eval); AUROC is kept alongside it. clip_norm/epsilon_final are only populated for
    DP-SGD rows, and CI columns may be absent for methods without groups — fill any
    missing column with NaN so the schema is stable.
    """
    df = pd.DataFrame(results)
    for col in _RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[_RESULT_COLUMNS]
    df.columns = _RESULT_HEADERS
    table_path = os.path.join(out_dir, "results_table.csv")
    df.to_csv(table_path, index=False)
    return df


# The family palette now lives in src.plotstyle and is COLOUR ONLY: the per-family marker
# shapes and line styles this dict used to carry were retired (VIZ_REDESIGN_TASK.md rule 2),
# because a reader should not have to learn a glyph legend to read a scatter. Re-exported
# here so the figure code below and compare_kaggle keep one import site for the palette.
FAMILY_COLOR = plotstyle.FAMILY_COLOR
MARKER = plotstyle.MARKER
DPSGD_METHOD = plotstyle.DPSGD_METHOD

# Figures in this module are drawn with the study's shared rcParams. Applied at import so
# that driving a plot function directly (as the tests do) produces the same styling as a
# full `make_figures` run; apply_style is idempotent.
plotstyle.apply_style()

# The width each figure's IN-AXES text is sized at. save_figure re-exports every figure at
# REPORT_WIDTH_IN (6.5in) with the same point sizes, so each relayout scales its axis
# labels, ticks, legends and point annotations by width/design_width — without which the
# two-panel figures' x-labels collide and the y-label grows into the subtitle.
EPSILON_DESIGN_WIDTH_IN = 11.0
MI_REPORT_DESIGN_WIDTH_IN = 9.0
DPSGD_DESIGN_WIDTH_IN = 10.5
PR_DESIGN_WIDTH_IN = 8.5
# The by-type chart is four bars in two groups. At PR_DESIGN_WIDTH_IN it stretched them
# into slabs with more gap than bar; narrower, the group spacing reads as grouping.
BY_TYPE_DESIGN_WIDTH_IN = 6.5

# The 95% CIs here are honestly huge: the cluster bootstrap resamples at session level
# and the test split holds only ~11 sessions, so effective n = 11 (MI-AUC ≈ ±0.15).
# Nothing is wrong with them — but drawn at full strength they read as geometry and
# drown the marks. Error bars appear ONLY on the report-tier figure (rule 4).
#
# The sentences that used to state this on-canvas — the CI source, the "MI-AUC below 0.5
# means the attack inverts" gloss, the seed-band note — are gone from the figures and live
# in Figure_Explainer.docx. They were three module constants here precisely because they
# were being retyped into five captions; the doc is now the one place they are written.


def _errorbar_kw(color):
    """Hairline, faded error bars that read as context and never compete with the marker."""
    return dict(elinewidth=0.6, capsize=0, ecolor=to_rgba(color, 0.35))


def _figure_caption(fig, text, fontsize=7, top_rect=1.0, chars_per_inch=16):
    """
    Place a footnote caption, wrapped to the figure width and given matching margin.

    Captions here accumulate clauses as findings are added (CI source + seed band +
    inverted-attack note), and a centred single line silently overflows BOTH edges once
    it outgrows the figure. Wrapping to the actual figure width, then reserving bottom
    space proportional to the resulting line count, keeps every clause on-canvas however
    long the text gets.

    chars_per_inch is measured, not guessed: at fontsize 7 italic matplotlib renders
    ~17 chars/inch worst-case ("x"-dense) and ~20 for realistic mixed-case text, so 16
    leaves margin and a centred line cannot overflow either edge.

    Idempotent: calling it again on the same figure REWRAPS the existing caption rather
    than stacking a second one on top. That is what lets save_figure re-run the layout at
    report width, where the same text needs more lines because the figure is narrower but
    the font is still 7pt.
    """
    width_chars = max(40, int(fig.get_size_inches()[0] * chars_per_inch))
    wrapped = textwrap.fill(" ".join(text.split()), width=width_chars)
    n_lines = wrapped.count("\n") + 1
    line_h = (fontsize + 3) / (fig.get_size_inches()[1] * fig.dpi)  # ~pts -> figure frac
    existing = getattr(fig, "_caption_artist", None)
    if existing is not None:
        existing.set_text(wrapped)
    else:
        fig._caption_artist = fig.text(0.5, 0.008, wrapped, ha="center", va="bottom",
                                       fontsize=fontsize, style="italic")
    fig.tight_layout(rect=(0, 0.012 + line_h * n_lines, 1, top_rect))
    return n_lines


# Report pages are ~6.5in of usable width, so the vector copies are scaled to that.
REPORT_WIDTH_IN = 6.5


def save_figure(fig, figures_dir, filename, dpi=150, relayout=None, **savefig_kw):
    """
    Save a figure as PNG (screen/slides) AND as a vector PDF sized for a report page.

    The PDF goes to a sibling `figures_pdf/` directory at REPORT_WIDTH_IN wide, keeping
    the original aspect ratio, so a figure tuned on screen drops into the report without
    rasterising. The figure's on-screen size is restored afterwards so callers can keep
    using it.

    `relayout` is a callback run AFTER the resize and before the PDF is written. Fonts are
    sized in points and do not shrink with the figure, so a 10.5in figure squeezed to 6.5in
    has text taking ~1.6x the relative width it had — enough to push axis labels off the
    canvas and captions into the x-label. Figures with hand-placed text pass a callback
    that re-runs their layout at the report size; figures whose text is all axis-managed
    do not need one.
    """
    png_path = os.path.join(figures_dir, filename)
    fig.savefig(png_path, dpi=dpi, **savefig_kw)
    print(f"Plot saved to {png_path}")

    pdf_dir = os.path.join(os.path.dirname(figures_dir.rstrip(os.sep)) or ".", "figures_pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, os.path.splitext(filename)[0] + ".pdf")
    w, h = fig.get_size_inches()
    try:
        fig.set_size_inches(REPORT_WIDTH_IN, REPORT_WIDTH_IN * h / w)
        if relayout is not None:
            relayout(fig)
        fig.savefig(pdf_path, format="pdf", **savefig_kw)
    finally:
        fig.set_size_inches(w, h)  # restore, so the PNG size isn't clobbered on re-save
        if relayout is not None:
            relayout(fig)  # and restore the on-screen layout to match
    return png_path


def load_seed_bands(out_dir_or_figures_dir):
    """
    Load output_seed_bands.csv (written by src.final_comparison) if it exists.

    Optional by design: the figures render fine without it, so a fresh checkout that has
    not run the band computation still produces every figure — it just omits the bands.
    """
    base = os.path.dirname(out_dir_or_figures_dir.rstrip(os.sep)) or "."
    for candidate in (os.path.join(out_dir_or_figures_dir, "output_seed_bands.csv"),
                      os.path.join(base, "output_seed_bands.csv")):
        if os.path.exists(candidate):
            return pd.read_csv(candidate)
    return None


def _band_stats(bands, method):
    """Per-σ min/max/mean of PR-AUC and mean effective MI across the noise draws."""
    sub = bands[bands["method"] == method]
    if sub.empty:
        return None
    g = sub.groupby("sigma")
    return pd.DataFrame({
        "sigma": g.size().index,
        "pr_lo": g["pr_auc"].min().to_numpy(),
        "pr_hi": g["pr_auc"].max().to_numpy(),
        "pr_mean": g["pr_auc"].mean().to_numpy(),
        "eff_mi_mean": g["eff_mi"].mean().to_numpy(),
        "n_seeds": g.size().to_numpy(),
    }).sort_values("sigma")


def pareto_frontier(df):
    """
    The non-dominated configs on (low effective MI-AUC, high PR-AUC).

    A config is DOMINATED if some other config is at least as private AND at least as
    accurate, and strictly better on one axis. Shading the dominated region makes the
    frontier read at a glance: anything inside the grey is a strictly worse choice than
    something on the boundary, whatever weighting of privacy vs utility you prefer.

    Returns the frontier rows sorted by effective MI-AUC (ascending).
    """
    sub = df[df["MI-AUC_eff"].notna() & df["PR-AUC"].notna()].copy()
    if sub.empty:
        return sub
    keep = []
    for i, row in sub.iterrows():
        dominated = (
            (sub["MI-AUC_eff"] <= row["MI-AUC_eff"]) & (sub["PR-AUC"] >= row["PR-AUC"]) &
            ((sub["MI-AUC_eff"] < row["MI-AUC_eff"]) | (sub["PR-AUC"] > row["PR-AUC"]))
        ).any()
        if not dominated:
            keep.append(i)
    return sub.loc[keep].sort_values("MI-AUC_eff")


def _pareto_staircase(front, x0, x1, y0):
    """
    The frontier as (xs, ys) step coordinates spanning [x0, x1].

    Moving right (leakier), the best achievable PR-AUC is non-decreasing, so the frontier
    is a staircase and everything below it is dominated. Computed once and drawn into every
    facet, which is what lets a reader see each family against the SAME boundary.
    """
    xs, ys = [x0], [y0]
    best = y0
    for _, r in front.iterrows():
        xs.append(r["MI-AUC_eff"]); ys.append(best)      # step across at the old level
        best = max(best, r["PR-AUC"])
        xs.append(r["MI-AUC_eff"]); ys.append(best)      # step up to this config
    xs.append(x1); ys.append(best)
    return xs, ys


def _draw_pareto(ax, xs, ys):
    """
    Draw the frontier staircase — a clean line, no fill.

    The single-panel version shaded everything below the frontier grey. Across four facets
    that tint would cover most of every panel and mute the marks it is meant to set off,
    and the boundary is legible on its own: it is the only grey line in the panel and the
    only one that steps.
    """
    ax.plot(xs, ys, color=plotstyle.AXIS, alpha=0.85, linewidth=0.9, zorder=1,
            label="Pareto frontier")


def _inverted_mask(sub):
    """Rows whose RAW MI-AUC fell below 0.5 (the attack inverts) → drawn with hollow markers."""
    return pd.to_numeric(sub["MI-AUC"], errors="coerce") < 0.5


def make_figures(df, figures_dir, models_dir, data, device, attack_prevalence,
                 baseline_model=None):
    """
    Regenerate all CARLA figures from a results table (shared with reeval).

    Ordering is editorial, and it is also the two tiers. The presentation tier leads —
    fig_headline.png (one point per family, "which do I pick?"), fig_sweeps.png (what σ
    actually does), utility_retention.png (utility kept at each privacy bar) — because
    between them they answer every question a reader arrives with. The report tier follows
    as the scrutiny material: privacy_utility_mi_report.png keeps every CI and the Pareto
    shading, dpsgd_grid.png shows the 6 DP-SGD runs as the 2-D (clip × noise) grid they
    actually are, and the PR-curve figures explain the recall ceiling.
    """
    baseline = df[df["Method"] == "Baseline"].iloc[0]
    # Optional: the five cross-figure operating points (src.final_comparison). None on a
    # checkout that has not run the decision view — every figure still renders, just
    # without its anchors.
    anchors = resolve_anchors(load_operating_points(figures_dir))

    # Deferred: src.summary_figures imports this module for save_figure, so importing it
    # at module scope would close a cycle.
    from src.summary_figures import (
        plot_headline, plot_sweeps, plot_utility_retention, write_retention_table)

    plot_headline(df, figures_dir, anchors=anchors)
    plot_sweeps(df, figures_dir)
    # Still generated, but REPORT APPENDIX rather than a slide — it is the precise answer to
    # "preserve utility as long as it is safe" and it needs utility_retention.csv beside it.
    plot_utility_retention(df, figures_dir, anchors=anchors)
    write_retention_table(df, os.path.dirname(figures_dir.rstrip(os.sep)) or ".")

    _plot_epsilon(df, figures_dir)
    _plot_mi_report(df, figures_dir, baseline, attack_prevalence, anchors=anchors)
    _plot_dpsgd_grid(df, figures_dir, attack_prevalence)
    _plot_pr_curves(models_dir, data, device, attack_prevalence, figures_dir,
                    baseline_model=baseline_model)
    _plot_pr_curves_by_type(models_dir, data, device, attack_prevalence, figures_dir,
                            baseline_model=baseline_model)
    _remove_retired_figures(figures_dir)


# Figures whose ROLE was taken over by another figure, not merely renamed. Deleted on every
# run so a stale copy can never be picked up for the deck by mistake — the names still
# appear in old notes, and a file that silently stops being regenerated is worse than one
# that is gone. Removed in both the PNG and the PDF tree.
_RETIRED_FIGURES = {
    "privacy_utility_mi.png": "split into _report.png",
    "privacy_utility_mi_slides.png": "superseded by fig_headline.png + fig_sweeps.png",
    "privacy_utility_summary.png": "superseded by fig_headline.png",
}


def _remove_retired_figures(figures_dir):
    pdf_dir = os.path.join(os.path.dirname(figures_dir.rstrip(os.sep)) or ".", "figures_pdf")
    for name, why in _RETIRED_FIGURES.items():
        for path in (os.path.join(figures_dir, name),
                     os.path.join(pdf_dir, os.path.splitext(name)[0] + ".pdf")):
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed retired {path} ({why})")


def _asym_yerr(sub, center_col, lo_col, hi_col):
    """(2, n) asymmetric error-bar array from a center column and its CI bounds.

    Clipped at 0 because percentile bootstrap CIs can occasionally fall on the wrong
    side of the point estimate (noted in the task) — a negative bar length would make
    matplotlib raise.
    """
    center = sub[center_col]
    return np.vstack([
        (center - sub[lo_col]).clip(lower=0).to_numpy(),
        (sub[hi_col] - center).clip(lower=0).to_numpy(),
    ])


def _sigma_label(v):
    """
    Compact σ text for point annotations.

    The densified sweep comes from logspace, so its σ are not round numbers — printing
    them raw gives '0.00521827'. Show 2 significant figures, scientific below 1e-2.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)  # 'per-feature', or the personalized per-vehicle map
    if f == 0:
        return "0"
    if f < 0.01:
        return f"{f:.1e}".replace("e-0", "e-")  # 1.8e-04 → 1.8e-4
    return f"{f:.2g}"


def _fmt_epsilon(v):
    """
    Compact ε text for the range chart's endpoint labels.

    ε here spans thirteen orders of magnitude (0.05 to 1.6e11), so one format cannot serve
    both ends: DP-SGD's budget has to read as a number a reader recognises ("0.05"), while
    Output's has to read as a magnitude, not a 12-digit integer nobody parses.
    """
    if not np.isfinite(v):
        return ""
    if v < 1000:
        return f"{v:.2g}"
    return f"{v:.1e}".replace("e+0", "e").replace("e+", "e")


def _split_by_sigma(sub):
    """
    Split rows into (numeric-σ ordered by σ, everything else).

    Ordering by σ — not by the metric — is what makes a connecting line honest: it traces
    the sweep's actual trajectory. Rows with no numeric σ ('per-feature') have no place on
    that ordering and are drawn as standalone markers.
    """
    s = pd.to_numeric(sub["σ / noise_mult"], errors="coerce")
    numeric = sub[s.notna()].assign(_sigma=s).sort_values("_sigma")
    return numeric, sub[s.isna()]


def _point_label(row):
    """
    Identity text for one point. A DP-SGD run is identified by its (clip, noise) cell —
    labelling it with σ alone would print a bare '1' and lose which clip it came from.
    """
    if row["Method"] == DPSGD_METHOD:
        return f"C={row['clip_norm']:g}/nm={float(row['σ / noise_mult']):g}"
    return _sigma_label(row["σ / noise_mult"])


# Row order for the ε chart: ascending by where the family's span sits, so the chart reads
# top-to-bottom as "formally meaningful → formally vacuous".
EPSILON_ROWS = [DPSGD_METHOD, "Input", "Output (last-layer)", "Output"]

# The conventional ceiling for a privacy budget anyone would defend in a paper. Nothing
# magic about 10 — it is the top of the range DP deployments actually cite, which is the
# point: it separates DP-SGD's ε from the other families' by orders of magnitude.
EPSILON_MEANINGFUL_MAX = 10.0

EPSILON_DESIGN_HEIGHT_IN = 4.2


def _epsilon_spans(df, methods=EPSILON_ROWS):
    """(method, lo, hi) per family over its finite ε, skipping families that have none."""
    spans = []
    for method in methods:
        eps = pd.to_numeric(df[df["Method"] == method]["ε"], errors="coerce")
        eps = eps[np.isfinite(eps) & (eps > 0)]   # log axis: a non-positive ε has no place
        if not eps.empty:
            spans.append((method, float(eps.min()), float(eps.max())))
    return spans


def _plot_epsilon(df, figures_dir):
    """
    The ε each family actually spends — one row per family on a log ε axis.

    This REPLACES a two-panel PR-AUC-vs-ε curve chart, and the replacement is a deletion
    more than a redesign. That figure's left panel was the sweeps' detection row with σ
    remapped to ε on the x-axis: same measurements, same shape, no new fact — and it
    invited exactly the cross-panel ε comparison its own subtitle had to warn against.
    Its right panel was six DP-SGD points already drawn, better, in dpsgd_grid.png.

    The one thing neither of those carries is the MAGNITUDE of ε, and it is the study's
    sharpest result: DP-SGD buys ε < 0.2, while the perturbation families' formal budgets
    run to 1e5 and beyond — numbers that are not "weak privacy" but no guarantee at all.
    A range chart says that in one glance, and the shaded ε ≤ 10 zone is the only thing a
    reader has to compare against. No utility axis here at all; utility is elsewhere.

    Personalized has no single analytic ε (it is a per-vehicle σ map), so it has no row.

    Takes only the table: with the utility axis gone this figure needs neither the baseline
    PR-AUC, nor the attack prevalence, nor the multi-seed bands the old curves were shaded
    with. `output_seed_bands.csv` is still computed and committed (src.final_comparison) —
    it is evidence that the output curves are a property of the mechanism rather than of
    one lucky draw — it simply has no figure consuming it now.
    """
    spans = _epsilon_spans(df)
    if not spans:
        print("  skip privacy_utility_epsilon.png (no finite \u03b5 in the table)")
        return

    fig, ax = plt.subplots(figsize=(EPSILON_DESIGN_WIDTH_IN, EPSILON_DESIGN_HEIGHT_IN))

    lo_all = min(lo for _, lo, _ in spans)
    hi_all = max(hi for _, _, hi in spans)
    x_lo, x_hi = lo_all / 6.0, hi_all * 6.0     # a little air at each end for the dots

    # The shaded zone is the reference the whole chart is read against, so it goes down
    # first and stays behind everything. It runs off the left edge because \u03b5 can be
    # arbitrarily small and "meaningful" has no lower bound \u2014 only a ceiling.
    ax.axvspan(x_lo, EPSILON_MEANINGFUL_MAX, color=plotstyle.family_color(DPSGD_METHOD),
               alpha=0.07, linewidth=0, zorder=0)
    ax.axvline(EPSILON_MEANINGFUL_MAX, color=plotstyle.AXIS, linewidth=1.0, alpha=0.9,
               zorder=1)
    ax.annotate(f"meaningful (\u03b5 \u2264 {EPSILON_MEANINGFUL_MAX:g})",
                (EPSILON_MEANINGFUL_MAX, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-5, -11), textcoords="offset points",
                fontsize=8, color=plotstyle.MUTED, ha="right", va="top")

    for y, (method, lo, hi) in enumerate(spans):
        color = plotstyle.family_color(method)
        # A bar, not a line: the span is a RANGE the family's configs cover, and a bar with
        # a dot at each end says "from here to here" without the two dots reading as two
        # measurements with something interpolated between them.
        ax.plot([lo, hi], [y, y], color=color, linewidth=3.0, solid_capstyle="round",
                alpha=0.55, zorder=3)
        ax.plot([lo, hi], [y, y], marker=plotstyle.MARKER, markersize=7, linestyle="none",
                color=color, zorder=4)
        # Endpoint values, outside the span so they never sit on the bar. These are the
        # numbers the figure exists to show, so they are printed, not read off the axis.
        ax.annotate(_fmt_epsilon(lo), (lo, y), textcoords="offset points", xytext=(-9, 0),
                    ha="right", va="center", fontsize=8, color=color)
        ax.annotate(_fmt_epsilon(hi), (hi, y), textcoords="offset points", xytext=(9, 0),
                    ha="left", va="center", fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    # Headroom above the first row for the zone tag \u2014 the top row is DP-SGD, which sits
    # INSIDE the shaded zone, so a tag placed at the canvas top lands on its own bar.
    ax.set_ylim(len(spans) - 0.5, -1.1)          # first row at the top
    ax.set_yticks(range(len(spans)))
    ax.set_yticklabels([SHORT_NAME.get(m, m) for m, _, _ in spans], fontsize=10)
    for tick, (method, _, _) in zip(ax.get_yticklabels(), spans):
        tick.set_color(plotstyle.family_color(method))
    ax.set_xlabel("\u03b5   (log scale, lower = stronger guarantee)")
    # Rows run horizontally, so the rule that helps is a vertical one \u2014 same override as
    # the timing figure.
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")

    title = "Formal privacy budget \u03b5 by defence family"
    caption = "Personalized omitted \u2014 per-vehicle \u03c3 map, no single analytic \u03b5"

    def relayout(f):
        plotstyle.scale_axes_text(f, EPSILON_DESIGN_WIDTH_IN)
        plotstyle.style_figure(f, title, caption=caption)

    relayout(fig)
    save_figure(fig, figures_dir, "privacy_utility_epsilon.png", relayout=relayout)
    plt.close(fig)


# Facet order for the report figure: the two families with an established or near-chance
# result first, then the two that collapse. Same left-to-right order as fig_sweeps so a
# reader moving between the two figures finds each family in the place they left it.
MI_REPORT_FACETS = ["Input", "Output", "Output (last-layer)", DPSGD_METHOD]


def _plot_mi_report(df, figures_dir, baseline, attack_prevalence, anchors=None):
    """
    The scrutiny figure: every config, every 95% CI, and the Pareto frontier \u2014 FACETED.

    One panel per family, not one panel for all of them. As a single axes this figure was a
    hairball: 40 points each carrying a horizontal AND a vertical 95% whisker, five hues
    interleaved through the same cluster, over a grey dominated-region wash. The bars are
    the point of the figure (see below) and they were exactly what made it unreadable \u2014
    every whisker crossed three other families' whiskers.

    Small multiples fix that without dropping anything: shared axes, so a panel's position
    still reads against every other panel; the SAME Pareto staircase in each, so "does this
    family reach the frontier?" is answerable per family instead of by tracing one line
    through five colours; and the baseline in every panel as the common reference point.

    This is the one figure that keeps its error bars (rule 4), precisely BECAUSE they are
    large \u2014 relocating the uncertainty here rather than dropping it study-wide is the
    point. The presentation figures state it in a phrase and send the reader here.

    POINTS ONLY. The sweep lines that used to connect each family's configs were parametric
    paths in \u03c3: both axes are metrics, so the line's direction encodes nothing a reader can
    name, and where leakage is non-monotonic in \u03c3 (Output) the path folded back and read as
    two branches of a curve that does not exist. The \u03c3-ordered view lives on fig_sweeps.png,
    where \u03c3 is an axis. Only the frontier may be a line here \u2014 it is a boundary, not a
    trajectory.

    X = effective MI-AUC = max(raw, 1\u2212raw), floored at 0.5 = chance (left, most private).
    Y = PR-AUC (headline utility). Bars are the 95% CI on each axis.
    """
    # Personalized is excluded here as it is everywhere else on a privacy\u2013utility axis, and
    # it is dropped BEFORE the frontier is computed \u2014 a config that is not drawn must not
    # silently define the boundary the drawn ones are judged against.
    drawn = df[~df["Method"].isin(FIGURE_EXCLUDED)]
    families = [m for m in MI_REPORT_FACETS if m in set(drawn["Method"])]
    if not families:
        print("  skip privacy_utility_mi_report.png (no families in the table)")
        return

    bpr, bmi = baseline["PR-AUC"], baseline["MI-AUC_eff"]
    base_row = baseline.to_frame().T

    ncols = 2
    nrows = int(np.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(MI_REPORT_DESIGN_WIDTH_IN, 7.2),
                             sharex=True, sharey=True, squeeze=False)
    flat = [ax for row in axes for ax in row]

    # Fix the window before the staircase is built \u2014 it spans the axes, so it needs the
    # limits, and on shared axes setting them once on the first panel sets them for all.
    flat[0].set_xlim(*MI_XLIM)
    flat[0].set_ylim(0.05, 0.75)
    y0 = flat[0].get_ylim()[0]
    front = pareto_frontier(drawn)
    stair = _pareto_staircase(front, *MI_XLIM, y0) if not front.empty else None

    for i, ax in enumerate(flat):
        if i >= len(families):
            ax.set_visible(False)          # odd family count: leave the cell empty
            continue
        method = families[i]
        color = plotstyle.family_color(method)

        if stair is not None:
            _draw_pareto(ax, *stair)
        # Reference rules in EVERY panel \u2014 each facet has to be readable on its own \u2014 but
        # tagged only in the first, since the axes are shared and four copies of the same
        # two words is the noise this pass exists to remove.
        first = i == 0
        plotstyle.baseline_line(ax, bpr, label=plotstyle.BASELINE_LABEL if first else None)
        plotstyle.chance_line(ax, 0.5, label=plotstyle.CHANCE_LABEL if first else None)

        # The baseline point sits in every panel: it is the thing each family is being
        # compared to, so making the reader carry it across from one panel defeats the facet.
        ax.errorbar([bmi], [bpr],
                    xerr=_asym_yerr(base_row, "MI-AUC_eff", "MI-AUC_eff_lo", "MI-AUC_eff_hi"),
                    yerr=_asym_yerr(base_row, "PR-AUC", "PR-AUC_lo", "PR-AUC_hi"),
                    marker=plotstyle.MARKER, linestyle="none",
                    color=plotstyle.BASELINE_COLOR, markersize=5.5, zorder=3,
                    **_errorbar_kw(plotstyle.BASELINE_COLOR))

        sub = drawn[(drawn["Method"] == method) & drawn["MI-AUC_eff"].notna()]
        if not sub.empty:
            ax.errorbar(
                sub["MI-AUC_eff"], sub["PR-AUC"],
                xerr=_asym_yerr(sub, "MI-AUC_eff", "MI-AUC_eff_lo", "MI-AUC_eff_hi"),
                yerr=_asym_yerr(sub, "PR-AUC", "PR-AUC_lo", "PR-AUC_hi"),
                marker=plotstyle.MARKER, linestyle="none", color=color, markersize=4.5,
                zorder=4, **_errorbar_kw(color),
            )
        # The panel title IS the series label, in the family's own colour \u2014 which is what
        # retires the legend the single-panel version needed.
        ax.set_title(SHORT_NAME.get(method, method), fontsize=11, fontweight="bold",
                     color=color, pad=8)
        if i % ncols == 0:
            ax.set_ylabel(f"Detection quality (PR-AUC @ {attack_prevalence:.0%})",
                          fontsize=9.5)
        if i >= len(families) - ncols:
            ax.set_xlabel(ATTACK_AXIS_LABEL, fontsize=9.5)

    title = "Privacy\u2013utility by configuration, per family"
    caption = EXCLUSION_NOTE

    def relayout(f):
        plotstyle.scale_axes_text(f, MI_REPORT_DESIGN_WIDTH_IN)
        plotstyle.style_figure(f, title, caption=caption)

    relayout(fig)
    save_figure(fig, figures_dir, "privacy_utility_mi_report.png", relayout=relayout)
    plt.close(fig)


def _plot_dpsgd_grid(df, figures_dir, attack_prevalence):
    """
    DP-SGD as the 2-D grid it actually is: noise_multiplier (rows) × clip_norm (cols).

    The 6 DP-SGD runs are a factorial grid, so drawing them as 3 crossing 2-point lines
    invented trends that were never measured. Two single-hue heatmaps (Blues = utility,
    Reds = privacy leak — never a diverging map, since neither metric has a meaningful
    midpoint here) show all 6 cells with their value and ε.
    """
    dpsgd = df[df["Method"] == DPSGD_METHOD].copy()
    if dpsgd.empty:
        print("  skip dpsgd_grid.png (no DP-SGD rows)")
        return
    dpsgd["_nm"] = pd.to_numeric(dpsgd["σ / noise_mult"], errors="coerce")
    dpsgd["_clip"] = pd.to_numeric(dpsgd["clip_norm"], errors="coerce")
    nms = sorted(dpsgd["_nm"].dropna().unique())
    clips = sorted(dpsgd["_clip"].dropna().unique())

    def grid(col):
        m = np.full((len(nms), len(clips)), np.nan)
        for i, nm in enumerate(nms):
            for j, c in enumerate(clips):
                cell = dpsgd[(dpsgd["_nm"] == nm) & (dpsgd["_clip"] == c)]
                if not cell.empty:
                    m[i, j] = cell[col].values[0]
        return m

    pr, mi, eps = grid("PR-AUC"), grid("MI-AUC"), grid("ε")

    fig, axes = plt.subplots(1, 2, figsize=(DPSGD_DESIGN_WIDTH_IN, 5.0))
    panels = [
        (axes[0], pr, "Blues", "Detection (PR-AUC)", "PR-AUC"),
        (axes[1], mi, "Reds", "Leakage (MI-AUC)", "MI-AUC"),
    ]
    for ax, matrix, cmap, panel_title, cbar_label in panels:
        # Heatmap cells ARE the grid, so the shared y-grid would only draw lines across
        # them; turn it off for these two axes only.
        ax.grid(False)
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        lo, hi = np.nanmin(matrix), np.nanmax(matrix)
        for i in range(len(nms)):
            for j in range(len(clips)):
                if np.isnan(matrix[i, j]):
                    continue
                # Flip the text to white once the cell is dark enough to swallow black.
                shade = (matrix[i, j] - lo) / (hi - lo) if hi > lo else 0.0
                txt = "white" if shade > 0.6 else "black"
                ax.text(j, i - 0.08, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt)
                if np.isfinite(eps[i, j]):
                    ax.text(j, i + 0.22, f"ε={eps[i, j]:.3g}", ha="center", va="center",
                            fontsize=7, color=txt, alpha=0.85)
        ax.set_xticks(range(len(clips)), [f"C={c:g}" for c in clips])
        ax.set_yticks(range(len(nms)), [f"nm={n:g}" for n in nms])
        ax.set_xlabel("clip norm C")
        ax.set_ylabel("noise multiplier")
        ax.set_title(panel_title, fontsize=10.5)
        cbar = fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
        cbar.outline.set_visible(False)

    # The colourmaps stay vivid \u2014 this is a 6-cell grid read by comparing cells, and the
    # values are printed in every one, so saturation here costs nothing and does the work.
    # The "levels, not contrasts" caveat (cell-to-cell gaps sit inside the \u00b10.1 CIs) is the
    # single most important thing to say about this figure and it is a sentence, so it is
    # in the doc, not on the grid.
    title = "DP-SGD across the clip \u00d7 noise grid"

    def relayout(f):
        plotstyle.scale_axes_text(f, DPSGD_DESIGN_WIDTH_IN)
        plotstyle.style_figure(f, title)

    relayout(fig)
    save_figure(fig, figures_dir, "dpsgd_grid.png", relayout=relayout)
    plt.close(fig)


def _representative_models(models_dir, device, baseline_model=None):
    """
    (label, method, model) triples for the PR-curve figure: baseline + one mid-noise model
    per family. Loaded from saved checkpoints (Output is derived from baseline via seeded
    noise, matching how the sweep produced it — no Output checkpoint is ever saved).
    Missing checkpoints are skipped so the figure still renders on partial model sets.

    The METHOD travels with each model rather than being recovered from list position: a
    caller colouring by `zip(reps, palette)` silently hands Input's colour to DP-SGD the
    moment one checkpoint is absent, which is exactly the partial-model-set case this
    function is built to tolerate.

    Families in FIGURE_EXCLUDED are filtered out here, at the one place the figure's model
    list is built, so the exclusion cannot be applied to some figures and forgotten on
    others. Personalized's checkpoint stays on disk and its row stays in the tables — the
    curve simply is not drawn, and on this figure it was near-exactly Input's anyway
    (PR-AUC 0.215 vs 0.210), which is the mechanical equivalence that got it excluded.
    """
    if baseline_model is None:
        baseline_model = load_carla_model(os.path.join(models_dir, "baseline.pth"), device)
    reps = [("Baseline", "Baseline", baseline_model)]

    candidates = [
        ("Input σ=0.1", "Input", os.path.join(models_dir, "input_sigma0.1.pth")),
        ("DP-SGD C=1.0/nm=1.0", DPSGD_METHOD,
         os.path.join(models_dir, "dpsgd_clip1.0_nm1.0.pth")),
        ("Personalized", "Personalized", os.path.join(models_dir, "personalized.pth")),
    ]
    reps.append(("Output σ=0.005", "Output", apply_output_noise(baseline_model, sigma=0.005)))
    for label, method, path in candidates:
        if method not in FIGURE_EXCLUDED and os.path.exists(path):
            reps.append((label, method, load_carla_model(path, device)))
    return reps


def _plot_pr_curves(models_dir, data, device, attack_prevalence, figures_dir,
                    baseline_model=None):
    """
    Full precision-recall curves for representative models, on the SAME
    prevalence-controlled eval set used for the reported PR-AUC.

    Lines are legitimate here and everywhere else in this module they are not: precision is
    a genuine function of recall, traced by sweeping the decision threshold, so the curve
    has a direction a reader can name. A horizontal dashed line marks prevalence = the
    no-skill floor (a random classifier's precision).
    """
    fig, ax = plt.subplots(figsize=(PR_DESIGN_WIDTH_IN, 5.6))

    reps = _representative_models(models_dir, device, baseline_model=baseline_model)

    prevalence = None
    curves = []  # (ap, artist) so the legend can be ordered by descending PR-AUC
    for label, method, model in reps:
        # Colour by family, looked up from the model's own method — never by position in
        # the list, which shifts when a checkpoint is missing.
        color = plotstyle.family_color(method)
        # Reuse the exact eval-set builder so the plotted curve matches the reported
        # PR-AUC (same seeded attacked subset, same prevalence).
        scores, labels, _, _, _ = _detection_eval_set(
            model, data["test_ds"], attack_prevalence=attack_prevalence, device=device
        )
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        prevalence = float(labels.mean())
        # steps-post: PR points are a step function; linear interpolation overstates it.
        line, = ax.plot(recall, precision, color=color, linewidth=1.8,
                        drawstyle="steps-post", label=f"{label}  (PR-AUC={ap:.3f})")
        curves.append((ap, line))

    handles = [ln for _, ln in sorted(curves, key=lambda t: t[0], reverse=True)]
    if prevalence is not None:
        noskill = ax.axhline(prevalence, color=plotstyle.AXIS, linestyle="--", linewidth=1.0,
                             label=f"no skill ({prevalence:.3f})")
        handles.append(noskill)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    # The legend stays \u2014 five curves that overlap for most of their length cannot be direct-
    # labelled \u2014 and stays ordered by descending PR-AUC, which makes it a ranking as well as
    # a key. That the shared cliff at recall 0.5 is the fuzzy/plateau split rather than a
    # property of any defence is the figure's paragraph in the doc.
    plotstyle.small_legend(ax, handles=handles, loc="upper right")

    title = "Precision\u2013recall by model"

    def relayout(f):
        plotstyle.scale_axes_text(f, PR_DESIGN_WIDTH_IN)
        plotstyle.style_figure(f, title)

    relayout(fig)
    save_figure(fig, figures_dir, "pr_curves.png", relayout=relayout)
    plt.close(fig)


def _plot_pr_curves_by_type(models_dir, data, device, attack_prevalence, figures_dir,
                            baseline_model=None):
    """
    Average precision per attack type — a grouped BAR chart, not PR curves.

    The finding is the plateau blind spot: fuzzy bursts (a whole sub-window replaced by
    noise) are near-perfectly detected, plateau freezes (one feature held at its mean) are
    near-invisible, and that gap — not any defence — is what caps the combined PR-AUC and
    puts the shared cliff in pr_curves.png.

    That finding is two numbers per model, and it was being carried by four full PR curves
    that actively obscured it. They formed two bands a reader had to mentally average, and
    each ended in a vertical drop to the no-skill floor at recall=1 — a PR artifact (the
    last threshold admits everything) that read as Output declining where it does no such
    thing. Bars are the honest geom for two levels of a categorical split: average
    precision has a true zero, the comparison IS the height ratio, and the plateau bars
    sitting a sliver above their no-skill line is the whole story, told once.

    Baseline vs Output σ=0.005 — the defence is drawn to show the blind spot is a property
    of the ATTACK, not something a defence introduces.
    """
    if baseline_model is None:
        baseline_model = load_carla_model(os.path.join(models_dir, "baseline.pth"), device)
    reps = [("Baseline", plotstyle.BASELINE_COLOR, baseline_model),
            ("Output σ=0.005", plotstyle.family_color("Output"),
             apply_output_noise(baseline_model, sigma=0.005))]

    # (model, type) -> AP, plus the per-type prevalence its no-skill line is drawn at.
    summary, type_prevalence = {}, {}
    for label, color, model in reps:
        scores, labels, normal_errors, _, attack_types = _detection_eval_set(
            model, data["test_ds"], attack_prevalence=attack_prevalence, device=device
        )
        N = len(normal_errors)
        normal_scores, attack_scores = scores[:N], scores[N:]
        for t, name in ATTACK_TYPE_NAMES.items():
            mask = attack_types == t
            if mask.sum() == 0:
                continue
            n_pos = int(mask.sum())
            y = np.concatenate([np.zeros(N), np.ones(n_pos)])
            s_vals = np.concatenate([normal_scores, attack_scores[mask]])
            summary[(label, name)] = average_precision_score(y, s_vals)
            # Each type is scored against ALL normals, so its no-skill floor is its OWN
            # prevalence — about half the combined 5%, not 5%. Computed, not assumed: a
            # rule drawn at the combined rate would sit above every plateau bar and invent
            # a "worse than chance" finding that is not in the data.
            type_prevalence[name] = n_pos / float(N + n_pos)

    names = [n for n in ATTACK_TYPE_NAMES.values() if any(k[1] == n for k in summary)]
    if not names:
        print("  skip pr_curves_by_type.png (no attack types in the eval set)")
        return

    fig, ax = plt.subplots(figsize=(BY_TYPE_DESIGN_WIDTH_IN, 5.0))
    x = np.arange(len(names))
    width = 0.3
    for k, (label, color, _) in enumerate(reps):
        offset = (k - (len(reps) - 1) / 2) * width
        vals = [summary.get((label, n), np.nan) for n in names]
        ax.bar(x + offset, vals, width, color=color, label=label, zorder=3)
        # Value labels: the plateau bars are an order of magnitude shorter than the fuzzy
        # ones, so their heights cannot be read off the axis and the number IS the point.
        for xi, v in zip(x + offset, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", va="bottom", fontsize=9,
                            color=plotstyle.INK, zorder=4)

    # One no-skill rule per type, each at that type's own prevalence and drawn only across
    # its own group — a single line spanning both groups would be wrong for at least one.
    # Tagged once, at the left edge of the first group, where it clears the bar tops: the
    # previous version printed the tag straight through a bar's value label.
    for xi, name in zip(x, names):
        p = type_prevalence.get(name)
        if p is not None:
            ax.hlines(p, xi - 0.5, xi + 0.5, color=plotstyle.AXIS, linestyle="--",
                      linewidth=1.0, zorder=2)
    if type_prevalence.get(names[0]) is not None:
        ax.annotate("no skill", (x[0] - 0.5, type_prevalence[names[0]]),
                    textcoords="offset points", xytext=(2, 4), ha="left", va="bottom",
                    fontsize=8, color=plotstyle.MUTED, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names], fontsize=10)
    ax.set_xlabel("Attack type")
    ax.set_ylabel("Average precision")
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, 1.08)
    plotstyle.small_legend(ax, loc="upper right")

    title = "Average precision by attack type"

    def relayout(f):
        plotstyle.scale_axes_text(f, BY_TYPE_DESIGN_WIDTH_IN)
        plotstyle.style_figure(f, title)

    relayout(fig)
    save_figure(fig, figures_dir, "pr_curves_by_type.png", relayout=relayout)
    plt.close(fig)
