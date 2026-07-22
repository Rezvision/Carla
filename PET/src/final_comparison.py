"""
Final comparison deliverables: paired-difference bootstrap + the decision view.

Usage (from repo root):
    python -m src.final_comparison --data_dir data/CARLA_processed --out_dir notebooks

Everything here is EVAL-ONLY — it re-scores a handful of existing checkpoints and
recomputes from the stored results table. No training.

Why paired differences (FINAL_COMPARISON_TASK.md §1): every method is evaluated on the
SAME test sessions, so two models' scores are positively correlated. Checking whether
their separate 95% CIs overlap throws that pairing away and is badly under-powered —
with only ~11 test sessions the marginal CIs are ~±0.1 and overlap almost everywhere,
even where the paired difference is unambiguous. Instead each bootstrap replicate
resamples sessions once, scores BOTH models on that replicate, and differences them.

Outputs:
    notebooks/paired_diffs.csv      — the key claims, each with a paired 95% CI
    notebooks/operating_points.csv  — one row per family: best config under a privacy bar
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.compare import (
    DEFAULT_ATTACK_PREVALENCE, DPSGD_METHOD,
    prepare_carla_data, load_carla_model,
)
from src.evaluate import (
    _detection_eval_set, effective_auc, paired_bootstrap_diff,
    reconstruction_errors, balance_members, balanced_subsample_indices,
)
from src.perturbation import apply_output_noise, apply_output_noise_last_layer

# The privacy bar for the decision view: effective MI-AUC at or below this counts as
# "acceptably private" (0.5 = chance, so 0.55 is a modest 5-point leakage allowance).
DEFAULT_MI_BAR = 0.55

# Output / Output-LL models are derived from baseline.pth by seeded noise, so a config is
# identified by (method, sigma) and rebuilt on demand rather than loaded from disk.
_DERIVED = {
    "Output": apply_output_noise,
    "Output (last-layer)": apply_output_noise_last_layer,
}


def _load_config_model(method, sigma, clip_norm, models_dir, device, baseline_model):
    """Rebuild the model for one results-table row (checkpoint load, or derived noise)."""
    if method == "Baseline":
        return baseline_model
    if method in _DERIVED:
        return _DERIVED[method](baseline_model, sigma=float(sigma))
    if method == "Input":
        return load_carla_model(os.path.join(models_dir, f"input_sigma{sigma}.pth"), device)
    if method == DPSGD_METHOD:
        return load_carla_model(
            os.path.join(models_dir, f"dpsgd_clip{float(clip_norm)}_nm{float(sigma)}.pth"), device)
    if method == "Personalized":
        return load_carla_model(os.path.join(models_dir, "personalized.pth"), device)
    raise ValueError(f"unknown method: {method}")


def model_scores(model, data, device, attack_prevalence=DEFAULT_ATTACK_PREVALENCE):
    """
    Per-model score vectors for both axes, aligned across models so they can be paired.

    Detection: `_detection_eval_set` is seeded, so every model sees the identical eval
    set (same attacked subset, same row order) — a prerequisite for paired differencing.
    MI: the same balanced member subset and non-member set as compare.evaluate_model, so
    the MI scores line up row-for-row across models too.

    Returns a dict of (scores, labels, groups) for 'detect' and 'mi'.
    """
    scores, labels, normal_errors, subset_indices, _ = _detection_eval_set(
        model, data["test_ds"], attack_prevalence=attack_prevalence, device=device)
    groups = np.asarray(data["test_groups"])
    detect_groups = np.concatenate([groups, groups[subset_indices]])

    n_non = len(data["mi_nonmember_ds"])
    members = balance_members(data["train_ds"], n_non, seed=42)
    member_idx = balanced_subsample_indices(len(data["train_ds"]), n_non, seed=42)
    member_err = reconstruction_errors(model, members, device=device)
    nonmember_err = reconstruction_errors(model, data["mi_nonmember_ds"], device=device)
    mi_scores = np.concatenate([-member_err, -nonmember_err])
    mi_labels = np.concatenate([np.ones(len(member_err)), np.zeros(len(nonmember_err))])
    mi_groups = np.concatenate([np.asarray(data["train_groups"])[member_idx],
                                np.asarray(data["mi_nonmember_groups"])])

    return {"detect": (scores, labels, detect_groups),
            "mi": (mi_scores, mi_labels, mi_groups)}


# ---------------------------------------------------------------------------
# Operating points: best config per family under the privacy bar
# ---------------------------------------------------------------------------

def choose_operating_points(df, mi_bar=DEFAULT_MI_BAR):
    """
    One row per family: the highest-PR-AUC config whose effective MI-AUC ≤ mi_bar.

    If a family has no config meeting the bar, fall back to its config CLOSEST to the bar
    (lowest effective MI-AUC) and flag it — a family that cannot reach the privacy bar at
    any noise level is itself a finding, so it is reported rather than dropped.
    """
    rows = []
    for method in ("Input", "Output", "Output (last-layer)", DPSGD_METHOD, "Personalized"):
        sub = df[(df["Method"] == method) & df["MI-AUC_eff"].notna()]
        if sub.empty:
            continue
        ok = sub[sub["MI-AUC_eff"] <= mi_bar]
        alt = None
        if not ok.empty:
            pick, meets, note = ok.loc[ok["PR-AUC"].idxmax()], True, ""
        else:
            pick, meets = sub.loc[sub["MI-AUC_eff"].idxmin()], False
            note = f"does not reach bar (eff-MI ≤ {mi_bar})"
            # "Closest to the bar" is the most private config, which for a family whose
            # privacy barely moves with σ can also be its WORST on utility (Input: σ=0.01
            # is 0.575/0.200 while σ=2.0 is 0.585/0.542 — a hair leakier, far more
            # accurate). Naming the family's own best-utility Pareto point stops the row
            # from misrepresenting the family when read on its own.
            alt = _family_best_utility_pareto(sub)
            if alt is not None and alt.equals(pick):
                alt = None
            if alt is not None:
                note += (f"; within-family Pareto point is {_config_label(alt)} "
                         f"(PR {float(alt['PR-AUC']):.3f}, eff-MI {float(alt['MI-AUC_eff']):.3f})")
        rows.append({
            "method": method,
            "config": _config_label(pick),
            "sigma": pick["σ / noise_mult"],
            "clip_norm": pick["clip_norm"],
            "eff_mi": float(pick["MI-AUC_eff"]),
            "pr_auc": float(pick["PR-AUC"]),
            "meets_bar": meets,
            "note": note,
            # The same alternative as machine-readable columns, not only as prose in the
            # note. The figures anchor on a family's representative config (src.anchors),
            # and parsing it back out of an English sentence would be absurd — so the
            # table carries it as data and the figures never hardcode a config.
            **_alt_columns(alt),
        })
    return pd.DataFrame(rows)


_ALT_COLUMNS = ("alt_config", "alt_sigma", "alt_clip_norm", "alt_eff_mi", "alt_pr_auc")


def _alt_columns(alt):
    """The within-family Pareto alternative as columns (all-NaN when there isn't one)."""
    if alt is None:
        return {c: np.nan for c in _ALT_COLUMNS}
    return {
        "alt_config": _config_label(alt),
        "alt_sigma": alt["σ / noise_mult"],
        "alt_clip_norm": alt["clip_norm"],
        "alt_eff_mi": float(alt["MI-AUC_eff"]),
        "alt_pr_auc": float(alt["PR-AUC"]),
    }


def _family_best_utility_pareto(sub):
    """
    Highest-PR-AUC config on a single family's own Pareto frontier.

    Restricting to the frontier first matters: it excludes configs that some other
    setting of the SAME family beats on both axes, so the row we name is a defensible
    alternative rather than merely the highest PR-AUC at any leakage.
    """
    from src.compare import pareto_frontier
    front = pareto_frontier(sub)
    if front.empty:
        return None
    return front.loc[front["PR-AUC"].idxmax()]


def _config_label(row):
    """Human-readable config id for a results-table row."""
    if row["Method"] == DPSGD_METHOD:
        return f"C={float(row['clip_norm']):g}, nm={float(row['σ / noise_mult']):g}"
    sigma = row["σ / noise_mult"]
    try:
        return f"σ={float(sigma):.5g}"
    except (TypeError, ValueError):
        return f"σ={sigma}"


# ---------------------------------------------------------------------------
# Paired differences for the key claims
# ---------------------------------------------------------------------------

def compute_paired_diffs(df, data, models_dir, device, out_dir,
                         attack_prevalence=DEFAULT_ATTACK_PREVALENCE, mi_bar=DEFAULT_MI_BAR,
                         n_boot=1000, seed=42):
    """
    Paired 95% CIs for the claims the write-up actually makes, into paired_diffs.csv:
      - each family's operating point vs Baseline (ΔPR-AUC and Δeff-MI)
      - Output sweet spot (σ≈0.00918) vs best DP-SGD (C=0.5, nm=1)
      - Output sweet spot vs best Input (σ=2.0)
    Only the handful of models involved are re-scored (minutes, no training).
    """
    ops = choose_operating_points(df, mi_bar=mi_bar)

    # Assemble the comparison list as (label, row_a, row_b): "a minus b".
    baseline_row = df[df["Method"] == "Baseline"].iloc[0]
    comparisons = []
    for _, op in ops.iterrows():
        row = df[(df["Method"] == op["method"]) &
                 (df["σ / noise_mult"].astype(str) == str(op["sigma"]))]
        if op["method"] == DPSGD_METHOD:
            row = row[pd.to_numeric(row["clip_norm"], errors="coerce") == float(op["clip_norm"])]
        comparisons.append((f"{op['method']} [{op['config']}] vs Baseline",
                            row.iloc[0], baseline_row))

    # The two head-to-heads named in the task.
    out_sweet = _closest_sigma_row(df, "Output", 0.00918)
    best_dpsgd = _best_pr_auc_row(df[df["Method"] == DPSGD_METHOD])
    best_input = _best_pr_auc_row(df[df["Method"] == "Input"])
    comparisons.append((f"Output sweet spot [{_config_label(out_sweet)}] vs "
                        f"best DP-SGD [{_config_label(best_dpsgd)}]", out_sweet, best_dpsgd))
    comparisons.append((f"Output sweet spot [{_config_label(out_sweet)}] vs "
                        f"best Input [{_config_label(best_input)}]", out_sweet, best_input))

    # The headline-deciding comparison: Output-LL σ=0.05 is the only config whose privacy
    # gain vs baseline is significant, while the Output sweet spot has the larger (but
    # unproven) gain. Comparing them to each other directly is what decides which one the
    # write-up should lead with, so it belongs in the table rather than being inferred by
    # eyeballing two separate vs-baseline rows.
    out_ll = _closest_sigma_row(df, "Output (last-layer)", 0.05)
    comparisons.append((f"Output-LL [{_config_label(out_ll)}] vs "
                        f"Output sweet spot [{_config_label(out_sweet)}]", out_ll, out_sweet))

    baseline_model = load_carla_model(os.path.join(models_dir, "baseline.pth"), device)

    # Score each distinct model once, then reuse across comparisons.
    cache = {}

    def scores_for(row):
        key = (row["Method"], str(row["σ / noise_mult"]), str(row["clip_norm"]))
        if key not in cache:
            print(f"  scoring {row['Method']} {_config_label(row)} …")
            model = _load_config_model(row["Method"], row["σ / noise_mult"],
                                       row["clip_norm"], models_dir, device, baseline_model)
            cache[key] = model_scores(model, data, device, attack_prevalence)
        return cache[key]

    records = []
    for label, row_a, row_b in comparisons:
        sa, sb = scores_for(row_a), scores_for(row_b)
        rec = {"comparison": label,
               "a": f"{row_a['Method']} [{_config_label(row_a)}]",
               "b": f"{row_b['Method']} [{_config_label(row_b)}]"}

        # ΔPR-AUC (utility): positive = A detects better than B.
        scores_a, labels, groups = sa["detect"]
        scores_b, _, _ = sb["detect"]
        mean, lo, hi = paired_bootstrap_diff(scores_a, scores_b, labels, groups,
                                             metric_fn=average_precision_score,
                                             n_boot=n_boot, seed=seed)
        rec.update(pr_auc_diff=mean, pr_auc_lo=lo, pr_auc_hi=hi,
                   pr_auc_significant=bool(lo > 0 or hi < 0))

        # Δeff-MI (privacy): positive = A leaks MORE than B.
        mscores_a, mlabels, mgroups = sa["mi"]
        mscores_b, _, _ = sb["mi"]
        mean, lo, hi = paired_bootstrap_diff(mscores_a, mscores_b, mlabels, mgroups,
                                             metric_fn=effective_auc,
                                             n_boot=n_boot, seed=seed)
        rec.update(eff_mi_diff=mean, eff_mi_lo=lo, eff_mi_hi=hi,
                   eff_mi_significant=bool(lo > 0 or hi < 0))
        records.append(rec)
        print(f"    ΔPR-AUC={rec['pr_auc_diff']:+.4f} [{rec['pr_auc_lo']:+.4f}, {rec['pr_auc_hi']:+.4f}]"
              f"   Δeff-MI={rec['eff_mi_diff']:+.4f} [{rec['eff_mi_lo']:+.4f}, {rec['eff_mi_hi']:+.4f}]")

    diffs = pd.DataFrame(records)
    diffs.to_csv(os.path.join(out_dir, "paired_diffs.csv"), index=False)

    # Attach each family's ΔPR-AUC vs baseline (with paired CI) to the operating points.
    ops = attach_paired_to_ops(ops, diffs)
    ops.to_csv(os.path.join(out_dir, "operating_points.csv"), index=False)
    return diffs, ops


def attach_paired_to_ops(ops, diffs):
    """
    Join each family's vs-Baseline paired difference (and its significance) onto the
    operating-points table.

    Split out from compute_paired_diffs so the decision table can be rebuilt from a saved
    paired_diffs.csv without re-running the bootstraps — the differences are the
    expensive part, the table is just formatting.
    """
    ops = ops.copy()
    lookup = {r["a"]: r for _, r in diffs.iterrows() if str(r["b"]).startswith("Baseline")}
    for col in ("pr_auc_diff_vs_baseline", "pr_auc_diff_lo", "pr_auc_diff_hi",
                "eff_mi_diff_vs_baseline", "eff_mi_diff_lo", "eff_mi_diff_hi"):
        ops[col] = np.nan
    # Significance = the paired 95% CI does not cross zero. Recorded per axis so the
    # decision table can mark a utility drop and a privacy gain independently.
    ops["pr_auc_significant"] = False
    ops["eff_mi_significant"] = False
    for i, op in ops.iterrows():
        r = lookup.get(f"{op['method']} [{op['config']}]")
        if r is None:
            continue
        ops.loc[i, "pr_auc_diff_vs_baseline"] = r["pr_auc_diff"]
        ops.loc[i, "pr_auc_diff_lo"] = r["pr_auc_lo"]
        ops.loc[i, "pr_auc_diff_hi"] = r["pr_auc_hi"]
        ops.loc[i, "pr_auc_significant"] = bool(r["pr_auc_significant"])
        ops.loc[i, "eff_mi_diff_vs_baseline"] = r["eff_mi_diff"]
        ops.loc[i, "eff_mi_diff_lo"] = r["eff_mi_lo"]
        ops.loc[i, "eff_mi_diff_hi"] = r["eff_mi_hi"]
        ops.loc[i, "eff_mi_significant"] = bool(r["eff_mi_significant"])
    return ops


# ---------------------------------------------------------------------------
# Multi-seed bands for the output families (eval-only)
# ---------------------------------------------------------------------------

def compute_output_seed_bands(df, data, models_dir, device, out_dir,
                              seeds=(42, 1, 2, 3, 4),
                              attack_prevalence=DEFAULT_ATTACK_PREVALENCE):
    """
    Re-derive the output families at every dense-sweep σ under several noise draws.

    Output / Output-LL models ARE just seeded Gaussian noise on baseline.pth, so a
    different seed is a different draw of the same mechanism — no training. Reporting the
    spread across draws answers the obvious objection to a single curve ("you got one
    lucky draw"), which matters most near the sweet spot where the curve turns.

    Writes notebooks/output_seed_bands.csv with one row per (method, sigma, seed).
    """
    baseline_model = load_carla_model(os.path.join(models_dir, "baseline.pth"), device)
    rows = []
    for method, noise_fn in _DERIVED.items():
        sigmas = sorted(pd.to_numeric(
            df[df["Method"] == method]["σ / noise_mult"], errors="coerce").dropna().unique())
        for sigma in sigmas:
            for seed in seeds:
                model = noise_fn(baseline_model, sigma=float(sigma), seed=int(seed))
                sc = model_scores(model, data, device, attack_prevalence)
                scores, labels, _ = sc["detect"]
                mi_scores, mi_labels, _ = sc["mi"]
                rows.append({
                    "method": method, "sigma": float(sigma), "seed": int(seed),
                    "pr_auc": float(average_precision_score(labels, scores)),
                    "eff_mi": float(effective_auc(mi_labels, mi_scores)),
                })
            last = rows[-len(seeds):]
            pr = [r["pr_auc"] for r in last]
            print(f"  {method:20s} σ={sigma:.5g}  PR-AUC {min(pr):.4f}–{max(pr):.4f} "
                  f"(spread {max(pr) - min(pr):.4f}) over {len(seeds)} draws")

    bands = pd.DataFrame(rows)
    bands.to_csv(os.path.join(out_dir, "output_seed_bands.csv"), index=False)
    return bands


def _closest_sigma_row(df, method, sigma):
    sub = df[df["Method"] == method].copy()
    s = pd.to_numeric(sub["σ / noise_mult"], errors="coerce")
    return sub.loc[(s - sigma).abs().idxmin()]


def _best_pr_auc_row(sub):
    return sub.loc[sub["PR-AUC"].idxmax()]


def _fmt_diff(mean, lo, hi, significant):
    """
    Format a paired difference, bolding it when the 95% CI excludes zero.

    Marking significance in the table itself matters here because several point
    estimates point the 'right' way while their intervals still straddle zero — the
    Output sweet spot's privacy gain being the headline example.
    """
    if pd.isna(mean):
        return "n/a"
    txt = f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    return f"**{txt}**" if significant else f"{txt} (n.s.)"


def markdown_table(ops, mi_bar=DEFAULT_MI_BAR):
    """Render operating points as a markdown table for the README / report."""
    lines = [
        "| Family | Config | eff-MI | PR-AUC | ΔPR-AUC vs baseline (95% paired CI) "
        "| Δeff-MI vs baseline (95% paired CI) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in ops.iterrows():
        d_pr = _fmt_diff(r.get("pr_auc_diff_vs_baseline"), r.get("pr_auc_diff_lo"),
                         r.get("pr_auc_diff_hi"), bool(r.get("pr_auc_significant", False)))
        d_mi = _fmt_diff(r.get("eff_mi_diff_vs_baseline"), r.get("eff_mi_diff_lo"),
                         r.get("eff_mi_diff_hi"), bool(r.get("eff_mi_significant", False)))
        flag = "" if r["meets_bar"] else " ⚠︎ does not reach bar"
        # Personalized's config is a per-vehicle σ map ("v1:0.10 | v2:0.10 | ..."), whose
        # pipes would otherwise be parsed as extra markdown table columns.
        config = str(r["config"]).replace("|", "\\|")
        lines.append(f"| {r['method']}{flag} | {config} | {r['eff_mi']:.3f} | "
                     f"{r['pr_auc']:.3f} | {d_pr} | {d_mi} |")
    # Footnote the full note for every flagged family. Without this the table shows only
    # "does not reach bar" and hides that the chosen row (most private) can be its
    # family's WORST on utility — the row would mislead when read on its own.
    flagged = [r for _, r in ops.iterrows() if not r["meets_bar"] and str(r.get("note", ""))]
    if flagged:
        lines.append("")
        for r in flagged:
            lines.append(f"- ⚠︎ **{r['method']}** — {r['note']}")

    lines.append("")
    lines.append(f"*Privacy bar: effective MI-AUC ≤ {mi_bar} (0.5 = chance). "
                 "**Bold** = 95% paired CI excludes zero; `(n.s.)` = interval crosses zero, "
                 "so the difference is not established. Negative Δeff-MI = less leakage. "
                 "A flagged family's row is its config CLOSEST to the bar, which is not "
                 "necessarily its best-utility choice — see the per-family notes above. "
                 "CIs are paired session bootstraps — both models are scored on the SAME "
                 "resampled sessions, not compared by checking whether separate CIs overlap.*")
    return "\n".join(lines)


_README_BEGIN = "<!-- BEGIN operating-points (generated by src.final_comparison) -->"
_README_END = "<!-- END operating-points -->"


def update_readme_table(readme_path, ops, mi_bar=DEFAULT_MI_BAR):
    """
    Insert (or refresh) the operating-points table in the README between HTML markers.

    Marker-delimited so re-running is idempotent — it replaces the previous block rather
    than appending a second copy, and any hand-written README prose around it is left
    untouched.
    """
    block = (f"{_README_BEGIN}\n"
             "## Decision view: best config per family under a privacy bar\n\n"
             f"{markdown_table(ops, mi_bar=mi_bar)}\n"
             f"{_README_END}")
    text = open(readme_path, encoding="utf-8").read() if os.path.exists(readme_path) else ""
    if _README_BEGIN in text and _README_END in text:
        head, rest = text.split(_README_BEGIN, 1)
        _, tail = rest.split(_README_END, 1)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return readme_path


if __name__ == "__main__":
    import sys
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/CARLA_processed")
    parser.add_argument("--out_dir", default="notebooks")
    parser.add_argument("--attack_prevalence", type=float, default=DEFAULT_ATTACK_PREVALENCE)
    parser.add_argument("--mi_bar", type=float, default=DEFAULT_MI_BAR,
                        help="Privacy bar for the decision view (effective MI-AUC).")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Noise draws per output-family σ for the min–max bands "
                             "(eval-only). 0 skips the band computation.")
    parser.add_argument("--skip_paired", action="store_true",
                        help="Only recompute the seed bands (paired_diffs.csv untouched).")
    parser.add_argument("--rebuild_tables", action="store_true",
                        help="Rebuild operating_points.csv + the README table from an "
                             "existing paired_diffs.csv (no model scoring, no bootstraps).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(os.path.join(args.out_dir, "results_table.csv"))

    if args.rebuild_tables:
        # Cheap path: reformat from cached differences. No dataset load, no scoring.
        diffs = pd.read_csv(os.path.join(args.out_dir, "paired_diffs.csv"))
        ops = attach_paired_to_ops(choose_operating_points(df, mi_bar=args.mi_bar), diffs)
        ops.to_csv(os.path.join(args.out_dir, "operating_points.csv"), index=False)
        update_readme_table("README.md", ops, mi_bar=args.mi_bar)
        print(markdown_table(ops, mi_bar=args.mi_bar))
        print(f"\nRebuilt {os.path.join(args.out_dir, 'operating_points.csv')} + README table")
        raise SystemExit(0)

    print("Loading sessions …")
    data = prepare_carla_data(args.data_dir)

    models_dir = os.path.join(args.out_dir, "models")

    if not args.skip_paired:
        print("\n=== Paired-difference bootstrap ===")
        diffs, ops = compute_paired_diffs(
            df, data, models_dir, device, args.out_dir,
            attack_prevalence=args.attack_prevalence, mi_bar=args.mi_bar, n_boot=args.n_boot)
        print(f"\nSaved {os.path.join(args.out_dir, 'paired_diffs.csv')}")
        print(f"Saved {os.path.join(args.out_dir, 'operating_points.csv')}")
        print("\n=== Operating points (decision view) ===")
        print(markdown_table(ops, mi_bar=args.mi_bar))
        update_readme_table("README.md", ops, mi_bar=args.mi_bar)
        print("\nREADME.md operating-points table updated")

    if args.seeds > 0:
        print(f"\n=== Output-family seed bands ({args.seeds} noise draws per σ) ===")
        compute_output_seed_bands(
            df, data, models_dir, device, args.out_dir,
            seeds=tuple([42] + list(range(1, args.seeds))),
            attack_prevalence=args.attack_prevalence)
        print(f"Saved {os.path.join(args.out_dir, 'output_seed_bands.csv')}")
