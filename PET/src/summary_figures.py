"""
The three presentation-tier figures — all eval-free, all redrawn from the tables.

Everything else in the study plots all 40 configs, which is the right detail for reading a
family's SHAPE and the wrong detail for choosing between families. These answer the
questions a reader actually arrives with:

  fig_headline.png      — "which defence should I pick?" One point per family in tradeoff
      space. POINTS ONLY: the axes are two metrics, so a line between two configs would be
      a parametric path in σ, and those fold back on themselves and read as nonsense.
  fig_sweeps.png        — "what does more noise actually do?" σ on the x-axis, so detection
      and leakage are each a genuine function of x and a line chart is honest.
  utility_retention.png — the supervisor's criterion ("preserve as much utility as possible
      for as long as it is safe"): for every privacy bar t, the best utility each family
      reaches subject to leakage ≤ t.

Both sources are notebooks/results_table.csv + notebooks/operating_points.csv, so they
regenerate without loading the dataset or a single checkpoint:

    python -m src.summary_figures --out_dir notebooks
"""
import os

import matplotlib
import numpy as np
import pandas as pd

from src import plotstyle
from src.anchors import (
    ATTACK_AXIS_LABEL, DETECTION_AXIS_LABEL, DPSGD_METHOD, FIGURE_EXCLUDED, MI_XLIM,
    SHORT_NAME, figure_families, load_operating_points, resolve_anchors,
)

# The retention sweep: 0.50 is "leak nothing beyond chance", 0.72 is past the baseline's
# own leakage and so is effectively "no privacy requirement at all".
RETENTION_BARS = (0.51, 0.55, 0.60, 0.65, 0.70)
# The sweep runs 0.50–0.72; the axis opens a hair left of 0.50 purely so the "chance"
# rule at t=0.5 is visible as a line rather than merged into the spine.
RETENTION_XLIM = (0.50, 0.72)
RETENTION_XPAD = 0.004

# The two qualification standards. "point" believes the point estimate; "ci" demands that
# the whole 95% interval clears the bar, which is what "demonstrably safe" has to mean.
# Only "point" is DRAWN now (rule 4: presentation figures state uncertainty in words, not
# in a second set of curves) — "ci" survives because it computes the caption's number.
_STANDARD_COL = {"point": "MI-AUC_eff", "ci": "MI-AUC_eff_hi"}

# Utility window for the headline. The x-window is MI_XLIM, shared with the report figure
# so a config sits at the same horizontal position in both; y opens below the collapsed
# families and above the baseline so every direct label has room outside the data.
HEADLINE_YLIM = (0.10, 0.70)

# The width the headline figure's label offsets are hand-tuned at (see scale_annotations).
SUMMARY_DESIGN_WIDTH_IN = 10.5


# ---------------------------------------------------------------------------
# Retention: best utility subject to leakage ≤ t
# ---------------------------------------------------------------------------

def retention_steps(df, method, standard="point"):
    """
    The step function "best PR-AUC this family reaches at privacy bar t", as breakpoints.

    Returns a DataFrame of (threshold, pr_auc, config) sorted by threshold, where each row
    is a bar at which the family's achievable utility JUMPS — i.e. a config that is better
    than everything qualifying at a stricter bar. Configs that never improve on a stricter
    config are dropped: they are real configs but they never define the curve.

    `standard` selects which leakage number has to clear the bar: the point estimate
    ("point") or the 95% upper bound ("ci", = demonstrably safe).

    Deliberately NOT extrapolated below the first breakpoint: a family with no qualifying
    config at bar t has no curve there at all, which is a different statement from "it
    retains 0% utility there" and must not be drawn as one.
    """
    col = _STANDARD_COL[standard]
    sub = df[(df["Method"] == method) & df[col].notna() & df["PR-AUC"].notna()]
    if sub.empty:
        return pd.DataFrame(columns=["threshold", "pr_auc", "config"])
    sub = sub.sort_values(col)
    rows, best = [], -np.inf
    for _, r in sub.iterrows():
        if r["PR-AUC"] > best:
            best = float(r["PR-AUC"])
            rows.append({"threshold": float(r[col]), "pr_auc": best,
                         "config": _config_label(r)})
    return pd.DataFrame(rows)


def retention_at(df, method, bar, standard="point"):
    """
    Best (pr_auc, config) for one family at one privacy bar; (nan, None) if none qualifies.

    Reads off retention_steps rather than re-filtering, so the table and the plotted curve
    can never disagree about what qualifies.
    """
    steps = retention_steps(df, method, standard=standard)
    ok = steps[steps["threshold"] <= bar]
    if ok.empty:
        return float("nan"), None
    last = ok.iloc[-1]
    return float(last["pr_auc"]), str(last["config"])


def retention_table(df, bars=RETENTION_BARS, methods=None):
    """
    The hand-to-the-supervisor table: for each bar, each family's best qualifying config
    and the % of baseline PR-AUC it retains, under BOTH standards.

    Wide (a column block per family) rather than long, because the question it answers is
    read across a row: "he names a bar — what does each family give me?"
    """
    baseline_pr = _baseline_pr_auc(df)
    methods = list(methods or _families(df))
    records = []
    for bar in bars:
        rec = {"privacy_bar_eff_mi": bar}
        for method in methods:
            key = SHORT_NAME.get(method, method)
            for standard in ("point", "ci"):
                pr, config = retention_at(df, method, bar, standard=standard)
                rec[f"{key}_config_{standard}"] = config
                rec[f"{key}_retained_{standard}_pct"] = (
                    float("nan") if np.isnan(pr) else 100.0 * pr / baseline_pr)
        records.append(rec)
    return pd.DataFrame(records)


def first_demonstrably_safe_bar(df, methods=None):
    """
    The strictest privacy bar at which ANY family has a demonstrably-safe config.

    The figure no longer draws — or captions — the "whole 95% interval clears the bar"
    standard, but the number is still what the write-up quotes for it, and it stays
    computed from the table rather than typed into the doc so the two cannot drift apart.
    Re-read it from here whenever the retention section is rebuilt.
    """
    firsts = []
    for method in (methods or _families(df)):
        steps = retention_steps(df, method, standard="ci")
        if not steps.empty:
            firsts.append(float(steps["threshold"].min()))
    return min(firsts) if firsts else float("nan")


def _families(df):
    """Every non-baseline family present, in the canonical figure order."""
    order = ["Input", "Output", "Output (last-layer)", DPSGD_METHOD, "Personalized"]
    present = set(df["Method"].unique())
    return [m for m in order if m in present]


def _baseline_pr_auc(df):
    base = df[df["Method"] == "Baseline"]
    if base.empty:
        raise ValueError("results table has no Baseline row to normalise retention against")
    return float(base.iloc[0]["PR-AUC"])


def _baseline_row(df):
    return df[df["Method"] == "Baseline"].iloc[0]


def _config_label(row):
    """Config id for a results-table row (mirrors final_comparison._config_label)."""
    if row["Method"] == DPSGD_METHOD:
        return f"C={float(row['clip_norm']):g}, nm={float(row['σ / noise_mult']):g}"
    sigma = row["σ / noise_mult"]
    try:
        return f"σ={float(sigma):.5g}"
    except (TypeError, ValueError):
        return f"σ={sigma}"


def anchor_config_label(row):
    """
    Config text for an operating point, trimmed to what fits beside a marker.

    Rule 3 says on-figure text is names and configs only, which makes this text the ONLY
    place a config is identified — so it has to be both short and unambiguous. σ is printed
    at 2 significant figures (the sweep is logspace, so raw values print as '0.0091809'),
    DP-SGD names its (clip, noise) cell because 'nm=1' alone loses the clip, and
    Personalized — whose config is a 45-character per-vehicle σ map — names its shape and
    leaves the values to operating_points.csv.
    """
    if row["method"] == DPSGD_METHOD:
        return f"C={float(row['clip_norm']):g}, nm={float(row['sigma']):g}"
    try:
        return f"σ={float(row['sigma']):.2g}"
    except (TypeError, ValueError):
        return "per-vehicle σ"


# ---------------------------------------------------------------------------
# Label placement
# ---------------------------------------------------------------------------

def _keep_inside_axes(fig, ax, annotations, pad_px=3.0):
    """
    Nudge annotations back inside the axes after layout.

    The headline figure direct-labels six points, and a label's real extent is only known
    once it is rendered — hand-tuned offsets that look fine at one figure size clip at
    another. So: draw, measure every label against the axes box, and shift any overhanging
    one back by exactly its overhang. Offsets stay hand-chosen for readability; this only
    guarantees the invariant.

    The overhang is measured in PIXELS but the position it corrects is in offset POINTS, so
    it has to be converted: applying pixels directly overshoots by dpi/72 (at the study's
    110 dpi, a label pushed off the right edge lands off the LEFT one instead).
    """
    fig.canvas.draw()
    box = ax.get_window_extent()
    px_to_pt = 72.0 / fig.dpi
    for ann in annotations:
        ext = ann.get_window_extent()
        dx = dy = 0.0
        # A label wider than the axes cannot satisfy both edges; prefer the left, so text
        # still starts where a reader looks for it rather than running off backwards.
        if ext.x1 > box.x1 - pad_px:
            dx = box.x1 - pad_px - ext.x1
        if ext.x0 + dx < box.x0 + pad_px:
            dx = box.x0 + pad_px - ext.x0
        if ext.y1 > box.y1 - pad_px:
            dy = box.y1 - pad_px - ext.y1
        if ext.y0 + dy < box.y0 + pad_px:
            dy = box.y0 + pad_px - ext.y0
        if dx or dy:
            ox, oy = ann.get_position()  # offset-points space
            ann.set_position((ox + dx * px_to_pt, oy + dy * px_to_pt))
    fig.canvas.draw()


def scale_annotations(annotations, base, width_in, design_width_in):
    """
    Rescale hand-placed labels for a figure being saved at a different width.

    Fonts and offsets are in POINTS — they do not shrink with the canvas — so the report
    copy at 6.5in would otherwise give every label ~1.6x its designed share of the figure
    and start colliding. `base` is the (fontsize, offset) each annotation was designed with,
    captured before any scaling so repeated calls never compound.
    """
    k = min(1.0, width_in / design_width_in)
    for ann, (fontsize, offset) in zip(annotations, base):
        ann.set_fontsize(fontsize * k)
        ann.set_position((offset[0] * k, offset[1] * k))
        # These carry an OFFSET as well as a size, so this function owns them entirely;
        # the flag keeps plotstyle.scale_axes_text from also scaling the size.
        ann._plotstyle_managed = True
    return k


# ---------------------------------------------------------------------------
# Figure 1: the headline tradeoff scatter
# ---------------------------------------------------------------------------

# Hand-placed label anchors: (offset_points, ha, va). The two bands of this scatter are far
# apart vertically, so labels go ABOVE the high-utility band and BELOW the collapsed one —
# no label ever crosses the gap that carries the figure's message. _keep_inside_axes then
# guarantees none of them leaves the axes whatever the figure size.
_LABEL_PLACEMENT = {
    "Output":              ((0, 13), "center", "bottom"),
    "Input":               ((0, 13), "center", "bottom"),
    "Output (last-layer)": ((0, 13), "center", "bottom"),
    "Baseline":            ((0, 13), "center", "bottom"),
    DPSGD_METHOD:          ((0, -14), "center", "top"),
}


def plot_headline(df, figures_dir, anchors=None, out_dir=None):
    """
    ONE point per family against the undefended baseline — the "which do I pick?" figure.

    POINTS ONLY, and that is the structural fix this figure exists for. Both axes are
    metrics, so any line drawn here is a parametric path in the hidden variable σ: it folds
    back on itself wherever a family's leakage is non-monotonic in σ (Output's does), and a
    folded path reads as two branches of a curve that does not exist. Sweeps belong on
    fig_sweeps.png, where σ is an actual axis.

    Colour is the only family encoding — one marker shape, no arrows, no verdict boxes. The
    movement from the baseline that arrows used to draw is legible without them: the
    baseline is the rightmost, highest point, so "left and slightly down" is the shape of a
    good defence and "left and far down" is the shape of an expensive one.

    Five points: four defence families plus the baseline. Personalized is excluded study-
    wide from the figures (see anchors.FIGURE_EXCLUDED) and keeps its row in the tables.
    """
    import matplotlib.pyplot as plt
    from src.compare import save_figure

    if anchors is None:
        anchors = resolve_anchors(load_operating_points(out_dir or figures_dir))
    if anchors is not None and not anchors.empty:
        anchors = anchors[~anchors["method"].isin(FIGURE_EXCLUDED)]
    baseline = _baseline_row(df)
    bx, by = float(baseline["MI-AUC_eff"]), float(baseline["PR-AUC"])

    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    plotstyle.chance_line(ax, 0.5, label="chance")
    ax.set_xlim(*MI_XLIM)
    ax.set_ylim(*HEADLINE_YLIM)

    annotations = []

    def point(method, x, y, color, label):
        ax.scatter([x], [y], marker=plotstyle.MARKER, s=190, color=color,
                   linewidths=0, zorder=5)
        offset, ha, va = _LABEL_PLACEMENT.get(method, ((0, 13), "center", "bottom"))
        annotations.append(ax.annotate(label, (x, y), textcoords="offset points",
                                       xytext=offset, ha=ha, va=va, fontsize=9.5,
                                       color=color, zorder=7))

    if anchors is not None and not anchors.empty:
        for _, a in anchors.iterrows():
            point(a["method"], float(a["eff_mi"]), float(a["pr_auc"]),
                  plotstyle.family_color(a["method"]),
                  f"{SHORT_NAME.get(a['method'], a['method'])}  {anchor_config_label(a)}")

    point("Baseline", bx, by, plotstyle.BASELINE_COLOR, "Baseline  (no defence)")

    ax.set_xlabel(ATTACK_AXIS_LABEL)
    ax.set_ylabel(DETECTION_AXIS_LABEL)

    # A noun phrase naming the figure, not a sentence stating its finding. The finding —
    # that Output reaches chance while keeping ~91% of detection, that only Output-LL's
    # privacy gain clears its paired CI, that Input's σ=2 retention is fuzzy-only — is the
    # figure's section in Figure_Explainer.docx. Nothing here should have to be READ.
    title = "Privacy–utility by defence family"

    base = [(a.get_fontsize(), a.get_position()) for a in annotations]

    def relayout(f):
        scale_annotations(annotations, base, f.get_size_inches()[0], SUMMARY_DESIGN_WIDTH_IN)
        plotstyle.scale_axes_text(f, SUMMARY_DESIGN_WIDTH_IN)
        plotstyle.style_figure(f, title)
        # Last: the labels can only be measured against the axes once the axes have been
        # given their final box by style_figure's tight_layout.
        _keep_inside_axes(f, ax, annotations)

    relayout(fig)
    save_figure(fig, figures_dir, "fig_headline.png", relayout=relayout)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: the σ sweeps as proper line charts
# ---------------------------------------------------------------------------

# Columns of fig_sweeps: the three families that HAVE a σ axis. DP-SGD is excluded on
# purpose — its runs are a (clip × noise) grid, not a σ sweep, so it has no x to plot
# against; the subtitle sends the reader to dpsgd_grid.png. Personalized is a single
# per-vehicle config, i.e. one point, which is not a sweep either.
SWEEP_FAMILIES = ["Input", "Output", "Output (last-layer)"]

# Six panels of log-scale ticks across one row: the width its text is sized at.
SWEEPS_DESIGN_WIDTH_IN = 12.0


def plot_sweeps(df, figures_dir):
    """
    What each family does as σ rises — as small multiples with σ on the x-axis.

    This is the figure that replaces the 2×2 tradeoff facets, and the axis swap is the whole
    point. In metric-vs-metric space a sweep is a parametric path: Output's leakage falls to
    chance and then rises again as the attack inverts, so its path folds and reads as two
    unrelated branches. Put σ on x and both metrics become genuine functions of it — the
    fold becomes what it actually is, a V in the leakage panel, and no line can branch.

    Rows are the two things σ trades: detection (top) and leakage (bottom). Each row shares
    a y-axis across all three columns, so "Input barely detects anything until σ=0.3" is
    readable as a height difference between panels rather than a tick-label comparison.
    """
    import matplotlib.pyplot as plt
    from src.compare import save_figure

    baseline = _baseline_row(df)
    present = [m for m in SWEEP_FAMILIES if m in set(df["Method"])]
    if not present:
        print("  skip fig_sweeps.png (no σ-sweep families in the table)")
        return

    # sharey="row" so "Input barely detects anything until σ=0.3" reads as a height
    # difference between panels rather than a tick-label comparison. sharex="col" because
    # each family's two panels span the SAME σ range, so the upper row's tick labels are
    # duplicate ink — matplotlib hides them and the axis is read once, at the bottom.
    fig, axes = plt.subplots(2, len(present), figsize=(SWEEPS_DESIGN_WIDTH_IN, 6.4),
                             sharey="row", sharex="col", squeeze=False)

    rows = [("PR-AUC", "Detection\n(PR-AUC)", float(baseline["PR-AUC"]), False),
            ("MI-AUC_eff", "Leakage\n(effective MI-AUC)", float(baseline["MI-AUC_eff"]), True)]

    for col, method in enumerate(present):
        color = plotstyle.family_color(method)
        sub = df[df["Method"] == method].copy()
        # Only numeric σ can sit on a σ axis. Input's 'per-feature' config has no single σ,
        # so it is dropped here rather than parked at an invented x — the caption says so.
        sub["_sigma"] = pd.to_numeric(sub["σ / noise_mult"], errors="coerce")
        sub = sub[sub["_sigma"].notna()].sort_values("_sigma")

        for row, (col_name, row_label, base_val, is_leakage) in enumerate(rows):
            ax = axes[row][col]
            # Tag the rules in the FIRST column only. The rule itself is drawn in every
            # panel (each one has to be readable on its own), but the panels share a y-axis
            # per row, so the same two words printed six times is the noise the strip is
            # meant to remove — one tag per row reads once and applies across.
            first = col == 0
            plotstyle.baseline_line(ax, base_val,
                                    label=plotstyle.BASELINE_LABEL if first else None)
            if is_leakage:
                plotstyle.chance_line(ax, 0.5, axis="y",
                                      label=plotstyle.CHANCE_LABEL if first else None)
            if not sub.empty:
                ax.plot(sub["_sigma"], sub[col_name], marker=plotstyle.MARKER,
                        markersize=5, linewidth=1.8, color=color, zorder=3)
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(SHORT_NAME.get(method, method), fontsize=11,
                             fontweight="bold", color=color, pad=8)
            else:
                ax.set_xlabel("noise level σ   (log scale)", fontsize=9.5)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9.5)

    # The reference lines now name themselves, so the subtitle that used to decode them is
    # redundant as well as prose. The effective-MI definition, the V where the attack
    # inverts, and Input's σ-less 'per-feature' config are all in the doc.
    title = "Detection and leakage as noise increases"

    def relayout(f):
        plotstyle.scale_axes_text(f, SWEEPS_DESIGN_WIDTH_IN)
        plotstyle.style_figure(f, title)

    relayout(fig)
    save_figure(fig, figures_dir, "fig_sweeps.png", relayout=relayout)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: utility retained vs privacy requirement
# ---------------------------------------------------------------------------

def plot_utility_retention(df, figures_dir, out_dir=None, anchors=None,
                           bars=RETENTION_BARS):
    """
    "Preserve as much utility as possible for as long as it is safe" — as a curve.

    Read it by picking your privacy bar t on the x-axis and reading UP: each family's curve
    gives the best utility it can still deliver while leaking no more than t. Curves step up
    rightward because relaxing the bar can only ever admit more configs. y is a genuine
    function of x here, so these lines are honest — unlike a path through tradeoff space.

    ONE standard is drawn (the point estimate). The second, "demonstrably safe" standard —
    the whole 95% interval clearing the bar — used to be a parallel set of dashed curves;
    at n=11 sessions it is so much stricter that it doubled the ink to make a single point.
    Both standards, per bar, are in utility_retention.csv, and the gap between them is the
    figure's paragraph in Figure_Explainer.docx.

    REPORT APPENDIX, not a slide. It answers the supervisor's "preserve utility as long as
    it is safe" question precisely, and that question needs the CSV beside it — which is
    exactly what a slide cannot carry. fig_headline and fig_sweeps lead the deck instead.
    """
    import matplotlib.pyplot as plt
    from src.compare import save_figure

    baseline_pr = _baseline_pr_auc(df)
    x_lo, x_hi = RETENTION_XLIM

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    plotstyle.baseline_line(ax, 100.0)
    plotstyle.chance_line(ax, 0.5)

    ends = []  # (method, colour, height where the family's curve leaves the axes)
    for method in figure_families(_families(df)):
        color = plotstyle.family_color(method)
        steps = retention_steps(df, method, standard="point")
        steps = steps[steps["threshold"] <= x_hi]
        if steps.empty:
            continue
        # steps-post: the curve holds its level from one breakpoint until the next config
        # qualifies, then jumps. The trailing point extends the last level to the right
        # edge; there is deliberately NO leading point at x_lo, so a family simply has no
        # curve left of its first qualifying config.
        xs = list(steps["threshold"]) + [x_hi]
        ys = [100.0 * v / baseline_pr for v in steps["pr_auc"]]
        ys = ys + [ys[-1]]
        line, = ax.plot(xs, ys, drawstyle="steps-post", linewidth=2.0, color=color,
                        zorder=4, label=SHORT_NAME.get(method, method))
        ax.plot(steps["threshold"], ys[:-1], marker=plotstyle.MARKER, markersize=5,
                linestyle="none", color=color, zorder=5)
        ends.append((method, color, line))

    # A LEGEND here, not the end labels this figure used to carry, and the standing rule
    # (plotstyle: direct-label when ends separate, legend when they converge) is why. These
    # curves converge in BOTH right-hand clusters — Output and Output-LL finish within
    # 0.1pp of each other at the top, Input and DP-SGD within a few points at the bottom —
    # so a right-edge label collides whichever way it is nudged, and the leader lines that
    # used to rescue it were themselves four extra strokes aimed into the data.
    plotstyle.small_legend(ax, handles=[ln for _, _, ln in ends], loc="lower right")

    ax.set_xlim(x_lo - RETENTION_XPAD, x_hi)
    ax.set_ylim(0, 118)
    ax.set_xlabel("Privacy requirement t = maximum tolerated attack success\n"
                  "←  stricter privacy requirement")
    ax.set_ylabel("Utility retained  (% of baseline PR-AUC)")
    # No secondary absolute-PR-AUC axis. It was the same scale relabelled, so it added a
    # second set of numbers for a reader to reconcile and never a second fact; the absolute
    # values are in utility_retention.csv, which is what gets handed over anyway.

    title = "Utility retained under each privacy requirement"

    def relayout(f):
        plotstyle.scale_axes_text(f, SUMMARY_DESIGN_WIDTH_IN)
        plotstyle.style_figure(f, title)

    relayout(fig)
    save_figure(fig, figures_dir, "utility_retention.png", relayout=relayout)
    plt.close(fig)


def write_retention_table(df, out_dir, bars=RETENTION_BARS):
    """Write utility_retention.csv — the grid to hand over when a bar is named."""
    table = retention_table(df, bars=bars)
    path = os.path.join(out_dir, "utility_retention.csv")
    table.to_csv(path, index=False)
    print(f"Saved {path}")
    return table


if __name__ == "__main__":
    import argparse
    import sys

    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    matplotlib.use("Agg")
    plotstyle.apply_style()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="notebooks")
    parser.add_argument("--figures_dir", default=None,
                        help="Defaults to <out_dir>/figures.")
    parser.add_argument("--attack_prevalence", type=float, default=0.05)
    parser.add_argument("--with_detail", action="store_true",
                        help="Also redraw the table-driven detail figures (MI report, ε, "
                             "DP-SGD grid). Still eval-free — the PR-curve figures need "
                             "checkpoints and are left to src.reeval.")
    args = parser.parse_args()

    figures_dir = args.figures_dir or os.path.join(args.out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    results = pd.read_csv(os.path.join(args.out_dir, "results_table.csv"))
    anchors = resolve_anchors(load_operating_points(args.out_dir))

    plot_headline(results, figures_dir, anchors=anchors)
    plot_sweeps(results, figures_dir)
    plot_utility_retention(results, figures_dir, anchors=anchors)
    write_retention_table(results, args.out_dir)

    if args.with_detail:
        from src.compare import _plot_epsilon, _plot_mi_report, _plot_dpsgd_grid
        baseline = results[results["Method"] == "Baseline"].iloc[0]
        _plot_epsilon(results, figures_dir)
        _plot_mi_report(results, figures_dir, baseline, args.attack_prevalence, anchors=anchors)
        _plot_dpsgd_grid(results, figures_dir, args.attack_prevalence)
