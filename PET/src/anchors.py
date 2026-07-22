"""
Operating points — the one config per family that the headline figure shows.

Every report-tier figure draws all 40 configs. That is the right amount of detail for
reading a family's shape, and the wrong amount for answering "which one would I pick?" —
so ONE config per family is designated the family's operating point, and fig_headline.png
is exactly those points and nothing else.

Anchors used to also be MARKED inside the detail figures, with a gold star inside a ring.
That is gone: colour is the only family encoding now (VIZ_REDESIGN_TASK.md rule 2), and a
second glyph layer on top of a dense marker field cost more legibility than the
cross-reference was worth. The cross-reference survives as a shared x-axis — the headline
and the report figure use the same window and wording (see ATTACK_AXIS_LABEL / MI_XLIM),
so a point picked off the headline sits at the same horizontal position in the report.

The points are read from notebooks/operating_points.csv (written by src.final_comparison),
never hardcoded here: the CSV is the single source of truth, so re-running the decision
view with a different privacy bar moves the operating point in every figure at once.
"""
import os

import numpy as np
import pandas as pd

DPSGD_METHOD = "DP-SGD (fixed-clip baseline)"

# Shared leakage axis: the headline and the report figure use the SAME label wording and
# the SAME x-window, so a config sits at the same horizontal position in both and the axis
# only has to be learned once.
#
# The wording is plain language rather than the metric name. "Effective MI-AUC, more
# private <- -> leakier" asked a reader to hold a metric definition in their head before
# they could read the figure at all; this says what the number MEANS at the one place it
# is used, and the max(AUC, 1-AUC) definition lives in the caption.
ATTACK_AXIS_LABEL = "Attack success   (0.5 = attacker learns nothing)"
DETECTION_AXIS_LABEL = "Detection quality (PR-AUC)"

# Hard floor just under chance (nothing is "more private than chance"), with room past the
# baseline's 0.704 for its label.
MI_XLIM = (0.49, 0.76)

# Families that are MEASURED but never DRAWN on a privacy–utility figure.
#
# Personalized is a single-client SIMULATION of per-user DP: one model, per-vehicle σ, no
# formal ε, and mechanically it IS input perturbation — which already has its own series.
# On a privacy–utility axis it therefore reads as a sixth independent defence when it is
# neither independent nor, at this scale, a defence: it is dominated, and its point sat in
# the leakiest corner implying a tradeoff it never actually made.
#
# It is NOT deleted. Its row stays in results_table.csv and operating_points.csv, its code
# stays in src/personalized.py, and it keeps a sentence in the write-up, because it is the
# bridge to the federated phase ("also prototyped per-vehicle noise as an FL bridge; see FL
# section"). Excluded from the figures, retained in the record.
FIGURE_EXCLUDED = ("Personalized",)

# The one clause the mi_report figure is allowed to keep, so a reader who notices the gap
# between the tables (six families) and the facets (four) is not left guessing.
EXCLUSION_NOTE = "Personalized excluded — single-client simulation, no formal ε"


def figure_families(methods):
    """Drop the measured-but-never-drawn families from an iterable of method names."""
    return [m for m in methods if m not in FIGURE_EXCLUDED]


# Short family names for direct labels — the full DP-SGD method name is 28 characters and
# will not fit next to a point.
SHORT_NAME = {
    "Input": "Input",
    "Output": "Output",
    "Output (last-layer)": "Output-LL",
    DPSGD_METHOD: "DP-SGD",
    "Personalized": "Personalized",
}

# One-phrase verdict per family. The significance clauses come from the paired CIs already
# joined onto the table (eff_mi_significant); the qualitative clauses are findings
# established by other figures — Input's high-σ "recovery" being fuzzy-only lives in
# pr_curves_by_type, DP-SGD's ε in the ε figure.
#
# These are no longer DRAWN. They used to sit in a boxed annotation beside every point,
# which put five hedged sentences on top of the data (rule 3); they now inform the prose in
# the headline figure's caption, and they stay here so that reasoning has one written home
# and so operating_points.csv keeps carrying it.
_VERDICT = {
    "Output (last-layer)": "privacy gain established ✓",
    "Output": "best on paper — gain not established",
    "Input": "recovery is fuzzy-only",
    DPSGD_METHOD: "only formal guarantee (ε<0.2)",
    "Personalized": "leakiest point; gain not established",
}


def load_operating_points(out_dir_or_figures_dir):
    """
    Load operating_points.csv, looking in a results dir or its figures/ subdirectory.

    Returns None when the file is absent: the figures must still render on a fresh
    checkout that has not run src.final_comparison, just without the anchors (same
    contract as compare.load_seed_bands).
    """
    base = os.path.dirname(str(out_dir_or_figures_dir).rstrip(os.sep)) or "."
    for candidate in (os.path.join(str(out_dir_or_figures_dir), "operating_points.csv"),
                      os.path.join(base, "operating_points.csv")):
        if os.path.exists(candidate):
            return pd.read_csv(candidate)
    return None


def resolve_anchors(ops):
    """
    One anchor row per family: the config the figures should mark, with its verdict.

    Which config? Normally the operating point itself — the family's best utility under
    the privacy bar. But `choose_operating_points` falls back, for a family that never
    reaches the bar, to its MOST PRIVATE config; and for a family whose leakage barely
    moves with σ that config is also close to its WORST on utility (Input σ=0.01 is
    eff-MI 0.575 / PR 0.200, against σ=2.0's 0.585 / 0.542 — a hair leakier for nearly
    triple the utility). Anchoring there would caricature the family.

    So the fallback row is kept only when the leakage it bought is actually demonstrated:
    when its Δeff-MI vs baseline has a paired CI excluding zero. Otherwise the family is
    represented by its own best-utility Pareto point (the alt_* columns). Concretely this
    keeps Output-LL at σ=0.05 (the one established privacy gain in the study) while moving
    Input to σ=2.0 and DP-SGD to C=0.5/nm=1.

    Returns a DataFrame with method/config/eff_mi/pr_auc/verdict/is_alt, or None.
    """
    if ops is None or ops.empty:
        return None
    rows = []
    for _, op in ops.iterrows():
        has_alt = "alt_config" in op.index and not pd.isna(op.get("alt_config"))
        established = bool(op.get("eff_mi_significant", False))
        use_alt = has_alt and not bool(op.get("meets_bar", False)) and not established
        rows.append({
            "method": op["method"],
            "config": op["alt_config"] if use_alt else op["config"],
            "eff_mi": float(op["alt_eff_mi"] if use_alt else op["eff_mi"]),
            "pr_auc": float(op["alt_pr_auc"] if use_alt else op["pr_auc"]),
            "sigma": op["alt_sigma"] if use_alt else op["sigma"],
            "clip_norm": op["alt_clip_norm"] if use_alt else op["clip_norm"],
            "verdict": _VERDICT.get(op["method"], ""),
            "is_alt": use_alt,
        })
    return pd.DataFrame(rows)
