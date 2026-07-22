"""
Re-evaluate saved checkpoints with the new PR-AUC utility metric — WITHOUT retraining.

Usage (from repo root):
    python -m src.reeval --data_dir data/CARLA_processed --out_dir notebooks
    python -m src.reeval --kaggle --data_path data/synthetic_telemetry_data.csv --out_dir notebooks

What it does (see PRAUC_TASK.md step 4):
  - Rebuilds the EXACT same splits as compare.py / compare_kaggle.py by calling their
    shared prepare_*_data helpers (same seeds/functions — no copy-pasted setup block).
  - Loads every checkpoint from notebooks/models (or notebooks/kaggle_models) and
    reconstructs the matching architecture before load_state_dict.
  - Re-derives the output-perturbation variants from baseline.pth with the same sigmas
    and seed (they are never saved as their own checkpoints — they are deterministic
    seeded noise on the baseline, so re-derivable at eval time).
  - Re-runs the shared evaluate_model to get PR-AUC (+ AUROC, F1, FPR, MI-AUC) and
    rewrites the results table + all figures.
  - ε cannot be recomputed for DP-SGD without retraining, so ε (and ε final-epoch) are
    carried over unchanged from the existing table by matching on (method, σ, clip_norm).
    No training happens anywhere in this script.

Crash safety: like the sweeps, partial results are flushed to *_partial.csv after every
model so a crash late in the re-eval never loses completed work.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

from src.perturbation import (
    apply_input_noise, apply_output_noise, apply_output_noise_last_layer,
    analytic_gaussian_epsilon, worst_case_sigma,
    output_noise_epsilon, output_last_layer_epsilon,
)
from src.train import train
from src.timing import run_meta, save_run_timing_meta, time_block
from src.timing_benchmark import BENCHMARK_EPOCHS
from src.compare import (
    DEFAULT_ATTACK_PREVALENCE, CLIP_RANGE, DELTA,
    OUTPUT_SIGMAS, DPSGD_CLIPS, DPSGD_NMS,
    prepare_carla_data, load_carla_model, _fresh_model,
    evaluate_model as evaluate_carla,
    build_results_table as build_carla_table, make_figures as make_carla_figures,
    _save_partial_results as _save_partial_carla,
)
from src.compare_kaggle import (
    OUTPUT_SIGMAS as KAGGLE_OUTPUT_SIGMAS,
    DPSGD_CLIPS as KAGGLE_DPSGD_CLIPS, DPSGD_NMS as KAGGLE_DPSGD_NMS,
    prepare_kaggle_data, load_kaggle_model, evaluate_model as evaluate_kaggle,
    build_results_table as build_kaggle_table, make_figures as make_kaggle_figures,
    _save_partial_results as _save_partial_kaggle,
)


# Output and Output-LL models are DERIVED from baseline.pth at eval time (seeded noise),
# so their σ sweep costs evaluation only — no training. compare.py's training path keeps
# its original 4 σ; here we densify to ~12 log-spaced values so the figures show real
# curves instead of 4-point pseudo-lines, and so the sweep reaches the σ where utility
# actually collapses (see PRESENTATION_FIGURE_TASK.md Tier 1a).
REEVAL_OUTPUT_SIGMAS = [float(s) for s in np.logspace(np.log10(1e-4), np.log10(5e-2), 12)]

# Tier 2 (opt-in, --extend_input_sweep): the trained input sweep stops at σ=0.3, before
# utility ever collapses, so the privacy–utility curve never shows its bend. These σ
# push into the degraded regime. They need TRAINING (input noise is applied to the data,
# not the weights), which is why they are behind a flag and off by default.
EXTENDED_INPUT_SIGMAS = [0.6, 1.0, 2.0]


# ---------------------------------------------------------------------------
# ε carry-over: match new rows to the existing table on (method, σ, clip_norm)
# ---------------------------------------------------------------------------

def _norm_scalar(v):
    """
    Canonicalise a σ / clip cell so a value written by the sweep matches the same value
    reconstructed here regardless of str-vs-float representation. Numeric strings collapse
    to repr(float(...)) ("0.0001", "1.0", "0.5"); non-numeric labels ("per-feature",
    "v1:0.104 | …") pass through as-is; empty / NaN / None all map to "".
    """
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return repr(float(s))
    except ValueError:
        return s


def _match_key(method, sigma, clip):
    return (str(method), _norm_scalar(sigma), _norm_scalar(clip))


def _read_old_table(table_path):
    """Load the existing results table (for ε carry-over). Empty frame if absent."""
    if not os.path.exists(table_path):
        print(f"  WARNING: {table_path} not found — ε values will be NaN")
        return pd.DataFrame()
    return pd.read_csv(table_path)


def _epsilon_lookup(old_df):
    """Map (method, σ, clip_norm) → (ε, ε final-epoch) from the existing table."""
    lookup = {}
    if old_df.empty:
        return lookup
    for _, row in old_df.iterrows():
        key = _match_key(row.get("Method"), row.get("σ / noise_mult"), row.get("clip_norm"))
        lookup[key] = (row.get("ε", float("nan")), row.get("ε (final epoch)", float("nan")))
    return lookup


def _merge_epsilon(results, lookup):
    """
    Attach carried-over ε / ε-final to each new result row (matched on method+σ+clip).
    Unmatched rows get NaN rather than a recomputed value — ε is authoritative in the
    original table (DP-SGD ε needs the training-time accountant, which we can't rerun).

    Rows that already carry an "epsilon" key are left alone: the output-perturbation
    families compute ε directly from baseline.pth (a closed form over the weights — no
    training involved), which is what lets the densified σ sweep have correct ε at σ
    values that never appear in the old table.
    """
    for row in results:
        if "epsilon" in row:
            row.setdefault("epsilon_final", float("nan"))
            continue
        key = _match_key(row["method"], row.get("sigma"), row.get("clip_norm"))
        eps, eps_final = lookup.get(key, (float("nan"), float("nan")))
        row["epsilon"] = eps
        row["epsilon_final"] = eps_final


def _personalized_sigma(old_df):
    """Recover the Personalized row's σ label (per-vehicle noise map) from the old table.

    The map is derived from the baseline model's per-vehicle MI at TRAIN time; we can't
    recompute the exact recorded label without re-running that estimation, so we reuse
    the value the sweep already recorded (it identifies the row and is display-only).
    """
    if not old_df.empty and (old_df["Method"] == "Personalized").any():
        return old_df.loc[old_df["Method"] == "Personalized", "σ / noise_mult"].values[0]
    return "personalized"


# ---------------------------------------------------------------------------
# Tier 2 — extended input sweep (the only path here that trains)
# ---------------------------------------------------------------------------

def _extend_input_sweep(add, data, models_dir, device, epochs, batch_size):
    """
    Train + evaluate input perturbation at EXTENDED_INPUT_SIGMAS, using exactly the
    settings compare.py's input sweep uses (same clip_range, same train(), same ε
    formula) so the new rows are directly comparable to the existing ones.

    A checkpoint already on disk is reused rather than retrained, so re-running this is
    cheap and interrupting it never wastes a completed model.
    """
    sensitivity = 2 * CLIP_RANGE  # bounded by clipping → valid local-DP ε
    for sigma in EXTENDED_INPUT_SIGMAS:
        label = str(sigma)
        save_path = os.path.join(models_dir, f"input_sigma{label}.pth")
        timing = None  # stays None when the checkpoint is reused: nothing was trained now,
                       # so the timing columns must be blank rather than invented
        if os.path.exists(save_path):
            print(f"  σ={label}: reusing existing checkpoint")
            model = load_carla_model(save_path, device)
        else:
            print(f"  σ={label}: training (no checkpoint on disk) …")
            noisy_train_ds = apply_input_noise(data["train_ds"], sigma=sigma, clip_range=CLIP_RANGE)
            model, timing = train(
                _fresh_model(), noisy_train_ds, data["val_ds"], save_path=save_path,
                epochs=epochs, batch_size=batch_size, device=device,
            )
        eps = analytic_gaussian_epsilon(worst_case_sigma(sigma), sensitivity=sensitivity, delta=DELTA)
        add("Input", label, model, epsilon=eps, timing=timing)


# ---------------------------------------------------------------------------
# CARLA re-eval
# ---------------------------------------------------------------------------

def reeval_carla(data_dir, out_dir, attack_prevalence=DEFAULT_ATTACK_PREVALENCE, device=None,
                 output_sigmas=None, extend_input_sweep=False, epochs=30, batch_size=256):
    """
    Re-evaluate the CARLA checkpoints and rewrite the table + figures.

    Args:
        output_sigmas: σ grid for the two output-perturbation families. Defaults to the
            densified REEVAL_OUTPUT_SIGMAS (evaluation-only, so this is free); pass
            compare.OUTPUT_SIGMAS to reproduce the original 4-point sweep.
        extend_input_sweep: Tier 2 — additionally TRAIN input models at
            EXTENDED_INPUT_SIGMAS so the input curve reaches the regime where utility
            collapses. Off by default; this is the only path in this script that trains,
            and it reuses any checkpoint already on disk.
        epochs, batch_size: training settings for extend_input_sweep (match compare.py).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)  # parity with compare.run's seeding
    if output_sigmas is None:
        output_sigmas = REEVAL_OUTPUT_SIGMAS

    models_dir = os.path.join(out_dir, "models")
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    table_path = os.path.join(out_dir, "results_table.csv")

    print("Loading sessions …")
    data = prepare_carla_data(data_dir)

    old_df = _read_old_table(table_path)
    eps_lookup = _epsilon_lookup(old_df)

    baseline_path = os.path.join(models_dir, "baseline.pth")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline_path}")
    baseline_model = load_carla_model(baseline_path, device)

    results = []
    meta = run_meta(device)  # device + hostname, identical on every row of this run
    eval_seconds_per_model = float("nan")  # timed once (TIMING_TASK.md part B)

    def add(method, sigma, model, clip_norm=None, epsilon=None, timing=None,
            derive_seconds=None):
        nonlocal eval_seconds_per_model
        if math.isnan(eval_seconds_per_model):
            with time_block(sync_device=device) as _t:
                metrics = evaluate_carla(model, data, device=device,
                                         attack_prevalence=attack_prevalence)
            eval_seconds_per_model = _t.seconds
            print(f"  [timing] one full evaluate_model() = {_t.seconds:.1f}s "
                  f"(bootstrap CIs dominate; measured once per run)")
        else:
            metrics = evaluate_carla(model, data, device=device,
                                     attack_prevalence=attack_prevalence)
        row = {"method": method, "sigma": sigma, **metrics, **meta}
        if clip_norm is not None:
            row["clip_norm"] = clip_norm
        if epsilon is not None:
            row["epsilon"] = epsilon  # computed here; _merge_epsilon won't overwrite it
        if timing is not None:
            row.update(timing.as_row())
        if derive_seconds is not None:
            row["derive_seconds"] = derive_seconds
        results.append(row)
        _save_partial_carla(results, out_dir)  # flush after every model (crash safety)
        print(f"  {method:32s} σ={str(sigma):<28} "
              f"PR-AUC={metrics['pr_auc']:.3f}  AUROC={metrics['auroc']:.3f}  "
              f"MI-AUC={metrics['mi_auc']:.3f}")

    print("\n=== Baseline ===")
    add("Baseline", None, baseline_model)

    print("\n=== Input Perturbation ===")
    for label in ("0.01", "0.05", "0.1", "0.3", "per-feature"):
        p = os.path.join(models_dir, f"input_sigma{label}.pth")
        if not os.path.exists(p):
            print(f"  skip input σ={label} (missing {p})")
            continue
        add("Input", label, load_carla_model(p, device))

    if extend_input_sweep:
        print("\n=== Input Perturbation (extended sweep — TRAINS models) ===")
        _extend_input_sweep(add, data, models_dir, device, epochs, batch_size)

    # ε for both output families is a closed form over the baseline weights + σ (Balle &
    # Wang analytic Gaussian mechanism), so every densified σ gets its exact ε with no
    # training and nothing to carry over.
    print(f"\n=== Output Perturbation (full model) — {len(output_sigmas)} σ re-derived from baseline ===")
    for sigma in output_sigmas:
        # Derived, not trained: record the ~0 derive cost explicitly (never blank).
        with time_block(sync_device=device) as _t:
            noisy = apply_output_noise(baseline_model, sigma=sigma)
        add("Output", sigma, noisy,
            epsilon=output_noise_epsilon(baseline_model, sigma=sigma, delta=DELTA),
            derive_seconds=_t.seconds)

    print(f"\n=== Output Perturbation (last-layer) — {len(output_sigmas)} σ re-derived from baseline ===")
    for sigma in output_sigmas:
        with time_block(sync_device=device) as _t:
            noisy = apply_output_noise_last_layer(baseline_model, sigma=sigma)
        add("Output (last-layer)", sigma, noisy,
            epsilon=output_last_layer_epsilon(baseline_model, sigma=sigma, delta=DELTA),
            derive_seconds=_t.seconds)

    print("\n=== DP-SGD ===")
    for clip in DPSGD_CLIPS:
        for nm in DPSGD_NMS:
            p = os.path.join(models_dir, f"dpsgd_clip{clip}_nm{nm}.pth")
            if not os.path.exists(p):
                print(f"  skip dpsgd clip={clip} nm={nm} (missing {p})")
                continue
            add("DP-SGD (fixed-clip baseline)", nm, load_carla_model(p, device), clip_norm=clip)

    print("\n=== Personalized DP ===")
    p = os.path.join(models_dir, "personalized.pth")
    if os.path.exists(p):
        add("Personalized", _personalized_sigma(old_df), load_carla_model(p, device))
    else:
        print(f"  skip personalized (missing {p})")

    _merge_epsilon(results, eps_lookup)  # carry ε over from the existing table

    df = build_carla_table(results, out_dir)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nTable saved to {table_path}")
    save_run_timing_meta(out_dir, eval_seconds_per_model, device)

    make_carla_figures(df, figures_dir, models_dir, data, device, attack_prevalence,
                       baseline_model=baseline_model)
    return df


# ---------------------------------------------------------------------------
# Kaggle re-eval
# ---------------------------------------------------------------------------

def reeval_kaggle(data_path, out_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    models_dir = os.path.join(out_dir, "kaggle_models")
    figures_dir = os.path.join(out_dir, "kaggle_figures")
    os.makedirs(figures_dir, exist_ok=True)
    table_path = os.path.join(out_dir, "kaggle_results_table.csv")

    print("Loading Kaggle telemetry data …")
    data = prepare_kaggle_data(data_path)
    n_test_anom = data["n_test_anom"]
    print(f"  Test: {data['n_test']} rows, {n_test_anom} anomalies")
    print(f"  NOTE: PR-AUC is computed on {n_test_anom} test positives with a 95% i.i.d. "
          f"row-bootstrap CI — expect wide intervals; treat overlapping gaps as noise.")

    old_df = _read_old_table(table_path)
    eps_lookup = _epsilon_lookup(old_df)

    baseline_path = os.path.join(models_dir, "baseline.pth")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline_path}")
    baseline_model = load_kaggle_model(baseline_path, device)

    results = []

    def add(method, sigma, model, clip_norm=None):
        metrics = evaluate_kaggle(model, data, device=device)
        row = {"method": method, "sigma": sigma, **metrics}
        if clip_norm is not None:
            row["clip_norm"] = clip_norm
        results.append(row)
        _save_partial_kaggle(results, out_dir)
        print(f"  {method:32s} σ={str(sigma):<12} "
              f"PR-AUC={metrics['pr_auc']:.3f}  AUROC={metrics['auroc']:.3f}  "
              f"MI-AUC={metrics['mi_auc']:.3f}")

    print("\n=== Baseline ===")
    add("Baseline", None, baseline_model)

    print("\n=== Input Perturbation ===")
    for label in ("0.01", "0.05", "0.1", "0.3", "per-feature"):
        p = os.path.join(models_dir, f"input_sigma{label}.pth")
        if not os.path.exists(p):
            print(f"  skip input σ={label} (missing {p})")
            continue
        add("Input", label, load_kaggle_model(p, device))

    print("\n=== Output Perturbation (full model) — re-derived from baseline ===")
    for sigma in KAGGLE_OUTPUT_SIGMAS:
        add("Output", sigma, apply_output_noise(baseline_model, sigma=sigma))

    print("\n=== Output Perturbation (last-layer) — re-derived from baseline ===")
    for sigma in KAGGLE_OUTPUT_SIGMAS:
        add("Output (last-layer)", sigma, apply_output_noise_last_layer(baseline_model, sigma=sigma))

    print("\n=== DP-SGD ===")
    for clip in KAGGLE_DPSGD_CLIPS:
        for nm in KAGGLE_DPSGD_NMS:
            p = os.path.join(models_dir, f"dpsgd_clip{clip}_nm{nm}.pth")
            if not os.path.exists(p):
                print(f"  skip dpsgd clip={clip} nm={nm} (missing {p})")
                continue
            add("DP-SGD (fixed-clip baseline)", nm, load_kaggle_model(p, device), clip_norm=clip)

    _merge_epsilon(results, eps_lookup)

    df = build_kaggle_table(results, out_dir)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nTable saved to {table_path}")

    make_kaggle_figures(df, figures_dir, models_dir, data, device, baseline_model=baseline_model)
    return df


if __name__ == "__main__":
    # Match compare.py's Windows/redirected-output UTF-8 guard for σ/ε characters.
    import sys
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle", action="store_true",
                        help="Re-evaluate the Kaggle tabular set instead of CARLA.")
    parser.add_argument("--data_dir", default="data/CARLA_processed",
                        help="CARLA processed data dir (CARLA mode).")
    parser.add_argument("--data_path", default="data/synthetic_telemetry_data.csv",
                        help="Kaggle CSV path (--kaggle mode).")
    parser.add_argument("--out_dir", default="notebooks")
    parser.add_argument("--attack_prevalence", type=float, default=DEFAULT_ATTACK_PREVALENCE,
                        help="Synthetic attack prevalence for the CARLA PR-AUC eval set "
                             "(default 0.05 = 5%%; ignored for --kaggle, which uses real labels).")
    parser.add_argument("--sparse_output_sweep", action="store_true",
                        help="Use compare.py's original 4 output σ instead of the densified "
                             "12-point sweep (CARLA only). The dense sweep is evaluation-only, "
                             "so it is on by default.")
    parser.add_argument("--extend_input_sweep", action="store_true",
                        help="Tier 2 (CARLA only, OFF by default): additionally TRAIN input "
                             f"models at σ={EXTENDED_INPUT_SIGMAS} so the input curve reaches "
                             "the regime where utility collapses. Reuses existing checkpoints; "
                             "this is the only option here that trains.")
    parser.add_argument("--timing_benchmark", action="store_true",
                        help="TIMING_TASK.md part C (CARLA only): instead of re-evaluating, "
                             "run the fixed-epoch cost benchmark for each family and write "
                             "timing_benchmark.csv + figures/timing_by_family.png + "
                             "timing_summary.md. Trains a few short runs into a scratch dir; "
                             "never touches notebooks/models.")
    parser.add_argument("--timing_retrospective", action="store_true",
                        help="TIMING_TASK.md part A: write timing_retrospective.csv from "
                             "checkpoint mtimes/logs (no compute). Only meaningful on the "
                             "machine that trained the models — mtimes do not survive a clone.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs for --extend_input_sweep (matches compare.py).")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Training batch size for --extend_input_sweep (matches compare.py).")
    args = parser.parse_args()

    if args.timing_retrospective:
        from src.timing_retro import run as run_retrospective
        run_retrospective(args.out_dir)
    elif args.timing_benchmark:
        # Cost benchmark, not a re-evaluation: it trains short fixed-length runs into a
        # scratch dir and writes its own outputs, leaving results_table.csv untouched.
        from src.timing_benchmark import run_benchmark
        run_benchmark(args.data_dir, args.out_dir,
                      epochs=BENCHMARK_EPOCHS, batch_size=args.batch_size)
    elif args.kaggle:
        reeval_kaggle(args.data_path, args.out_dir)
    else:
        reeval_carla(
            args.data_dir, args.out_dir, attack_prevalence=args.attack_prevalence,
            output_sigmas=OUTPUT_SIGMAS if args.sparse_output_sweep else None,
            extend_input_sweep=args.extend_input_sweep,
            epochs=args.epochs, batch_size=args.batch_size,
        )
