"""
Tests for the final-comparison deliverables (FINAL_COMPARISON_TASK.md):
Pareto frontier / operating points, the vector-PDF export, seed-band plumbing, and the
colourblind safety of the family palette.

Everything here is pure computation or headless plotting — no checkpoints, no dataset.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt

from src.compare import (
    FAMILY_COLOR, MARKER, DPSGD_METHOD, REPORT_WIDTH_IN,
    pareto_frontier, save_figure, load_seed_bands, _band_stats,
)
from src.final_comparison import (
    choose_operating_points, markdown_table, DEFAULT_MI_BAR,
)


_COLUMNS = ["Method", "σ / noise_mult", "clip_norm", "PR-AUC", "MI-AUC_eff"]


def _table(rows):
    """Minimal results-table frame: (method, sigma, clip, pr_auc, eff_mi).

    Always carries the full column set — a real results table has the columns even when a
    filter leaves zero rows, and that is the empty case the code must handle.
    """
    return pd.DataFrame(
        [dict(zip(_COLUMNS, r)) for r in rows], columns=_COLUMNS)


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def test_pareto_frontier_keeps_only_non_dominated():
    # B is dominated by A (A is both more private and more accurate).
    df = _table([("Output", 0.1, np.nan, 0.60, 0.55),    # A — on frontier
                 ("Output", 0.2, np.nan, 0.50, 0.60),    # B — dominated by A
                 ("Input", 0.3, np.nan, 0.70, 0.70)])    # C — leakier but best utility
    front = pareto_frontier(df)
    prs = sorted(front["PR-AUC"].tolist())
    assert prs == [0.60, 0.70], "only the non-dominated configs should survive"


def test_pareto_frontier_sorted_by_privacy():
    df = _table([("Output", 0.1, np.nan, 0.70, 0.70),
                 ("Output", 0.2, np.nan, 0.60, 0.55),
                 ("Output", 0.3, np.nan, 0.50, 0.51)])
    front = pareto_frontier(df)
    assert list(front["MI-AUC_eff"]) == sorted(front["MI-AUC_eff"])


def test_pareto_frontier_handles_empty_and_nan():
    assert pareto_frontier(_table([])).empty
    df = _table([("Output", 0.1, np.nan, np.nan, 0.5)])
    assert pareto_frontier(df).empty


# ---------------------------------------------------------------------------
# Operating points (decision view)
# ---------------------------------------------------------------------------

def test_operating_point_picks_best_utility_under_the_bar():
    df = _table([
        ("Output", 0.001, np.nan, 0.61, 0.70),   # too leaky
        ("Output", 0.009, np.nan, 0.55, 0.51),   # under bar, best utility of those
        ("Output", 0.05, np.nan, 0.09, 0.52),    # under bar but worse utility
    ])
    ops = choose_operating_points(df, mi_bar=0.55)
    row = ops[ops["method"] == "Output"].iloc[0]
    assert row["pr_auc"] == 0.55
    assert bool(row["meets_bar"]) is True
    assert row["note"] == ""


def test_operating_point_flags_family_that_cannot_reach_bar():
    """A family with no config under the bar is reported (closest one) and flagged, not
    silently dropped — 'cannot reach the bar at any σ' is itself the finding."""
    df = _table([("Input", 0.01, np.nan, 0.20, 0.575),
                 ("Input", 2.0, np.nan, 0.54, 0.585)])
    ops = choose_operating_points(df, mi_bar=0.55)
    row = ops[ops["method"] == "Input"].iloc[0]
    assert bool(row["meets_bar"]) is False
    assert "does not reach bar" in row["note"]
    assert row["eff_mi"] == 0.575, "should fall back to the config closest to the bar"


def test_operating_points_label_dpsgd_by_clip_and_noise():
    df = _table([(DPSGD_METHOD, 1.0, 0.5, 0.22, 0.54)])
    ops = choose_operating_points(df, mi_bar=0.55)
    assert ops.iloc[0]["config"] == "C=0.5, nm=1"


def test_markdown_table_renders_rows_and_bar_note():
    df = _table([("Output", 0.009, np.nan, 0.55, 0.51),
                 ("Input", 0.01, np.nan, 0.20, 0.575)])
    ops = choose_operating_points(df, mi_bar=DEFAULT_MI_BAR)
    md = markdown_table(ops, mi_bar=DEFAULT_MI_BAR)
    assert "| Family |" in md and "Output" in md and "Input" in md
    assert "does not reach bar" in md
    assert str(DEFAULT_MI_BAR) in md


def test_markdown_table_marks_significance_from_paired_cis():
    """A CI that excludes zero is bolded; one that straddles zero is marked (n.s.) —
    the Output sweet spot's privacy gain is exactly the latter case."""
    df = _table([("Output", 0.009, np.nan, 0.55, 0.51)])
    ops = choose_operating_points(df, mi_bar=DEFAULT_MI_BAR)
    ops["pr_auc_diff_vs_baseline"], ops["pr_auc_diff_lo"], ops["pr_auc_diff_hi"] = -0.054, -0.070, -0.028
    ops["pr_auc_significant"] = True                       # CI excludes zero
    ops["eff_mi_diff_vs_baseline"], ops["eff_mi_diff_lo"], ops["eff_mi_diff_hi"] = -0.144, -0.247, 0.073
    ops["eff_mi_significant"] = False                      # CI crosses zero
    md = markdown_table(ops, mi_bar=DEFAULT_MI_BAR)
    assert "**-0.054 [-0.070, -0.028]**" in md, "significant ΔPR-AUC must be bold"
    assert "-0.144 [-0.247, +0.073] (n.s.)" in md, "zero-crossing Δeff-MI must be marked n.s."
    assert "**-0.144" not in md, "a non-significant difference must not be bolded"


# ---------------------------------------------------------------------------
# Vector PDF export
# ---------------------------------------------------------------------------

def test_save_figure_writes_png_and_report_sized_pdf(tmp_path):
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot([0, 1], [0, 1])
    save_figure(fig, str(figures_dir), "demo.png")
    plt.close(fig)

    assert (figures_dir / "demo.png").exists()
    pdf = tmp_path / "figures_pdf" / "demo.pdf"
    assert pdf.exists() and pdf.stat().st_size > 0
    assert pdf.read_bytes()[:4] == b"%PDF", "must be a real vector PDF"


def test_save_figure_restores_original_size(tmp_path):
    """Resizing for the PDF must not shrink the figure for subsequent PNG saves."""
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot([0, 1], [0, 1])
    save_figure(fig, str(figures_dir), "demo.png")
    assert tuple(fig.get_size_inches()) == (13, 5)
    plt.close(fig)


def test_report_width_is_page_sized():
    assert 5.0 <= REPORT_WIDTH_IN <= 7.5


# ---------------------------------------------------------------------------
# Seed bands
# ---------------------------------------------------------------------------

def test_load_seed_bands_absent_returns_none(tmp_path):
    """Figures must still render on a checkout that never computed the bands."""
    d = tmp_path / "figures"
    d.mkdir()
    assert load_seed_bands(str(d)) is None


def test_band_stats_reduces_seeds_to_min_max_mean():
    bands = pd.DataFrame([
        {"method": "Output", "sigma": 0.01, "seed": s, "pr_auc": pr, "eff_mi": mi}
        for s, pr, mi in [(42, 0.50, 0.60), (1, 0.54, 0.62), (2, 0.52, 0.61)]
    ])
    stats = _band_stats(bands, "Output")
    assert len(stats) == 1
    row = stats.iloc[0]
    assert row["pr_lo"] == pytest.approx(0.50)
    assert row["pr_hi"] == pytest.approx(0.54)
    assert row["pr_mean"] == pytest.approx(0.52)
    assert row["eff_mi_mean"] == pytest.approx(0.61)
    assert row["n_seeds"] == 3


def test_band_stats_unknown_method_returns_none():
    bands = pd.DataFrame([{"method": "Output", "sigma": 0.01, "seed": 42,
                           "pr_auc": 0.5, "eff_mi": 0.6}])
    assert _band_stats(bands, "Input") is None


# ---------------------------------------------------------------------------
# Colourblind safety of the family palette
# ---------------------------------------------------------------------------

_DEUTERANOPIA = np.array([[0.625, 0.375, 0.0],
                          [0.700, 0.300, 0.0],
                          [0.000, 0.300, 0.7]])


def _simulate_deuteranopia(colour):
    return np.clip(_DEUTERANOPIA @ np.array(to_rgb(colour)), 0, 1)


def test_family_colours_remain_distinct_under_deuteranopia():
    """
    Regression for the palette, and it matters MORE than it used to.

    Colour is now the only family encoding — the per-family marker shapes that used to be
    the backup channel are gone (VIZ_REDESIGN_TASK.md rule 2) — so a pair that collapses
    under deuteranopia is no longer merely inconvenient, it is unreadable. ('purple' for
    Personalized collapsed onto DP-SGD's seagreen at distance 0.08, hence magenta.)
    """
    names = list(FAMILY_COLOR)
    worst, pair = 1e9, None
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.linalg.norm(_simulate_deuteranopia(FAMILY_COLOR[a])
                                     - _simulate_deuteranopia(FAMILY_COLOR[b])))
            if d < worst:
                worst, pair = d, (a, b)
    assert worst >= 0.15, f"{pair[0]} and {pair[1]} are indistinguishable ({worst:.3f})"


def test_there_is_exactly_one_marker_shape():
    """
    The inverse of the old 'markers are unique' test, and a deliberate reversal.

    Five marker shapes across the figures meant a reader had to learn a glyph legend before
    reading a scatter, and shape was carrying facts (inverted attack, operating point) that
    belong in prose. There is now one shape, and colour alone says which family a point is.
    """
    assert MARKER == "o"


def test_baseline_is_achromatic_and_not_a_family_colour():
    """The undefended baseline is a REFERENCE, so it must never read as a sixth family."""
    r, g, b = to_rgb(FAMILY_COLOR["Baseline"])
    assert max(r, g, b) - min(r, g, b) < 0.05, "baseline colour must be grey"
    families = {k: v for k, v in FAMILY_COLOR.items() if k != "Baseline"}
    assert FAMILY_COLOR["Baseline"] not in families.values()


def test_markdown_table_escapes_pipes_in_config():
    """Personalized's per-vehicle σ map contains '|', which would otherwise be parsed as
    extra markdown columns and shift every later cell."""
    df = _table([("Personalized", "v1:0.104 | v2:0.097", np.nan, 0.215, 0.671)])
    ops = choose_operating_points(df, mi_bar=DEFAULT_MI_BAR)
    md = markdown_table(ops, mi_bar=DEFAULT_MI_BAR)
    row = [l for l in md.splitlines() if "Personalized" in l][0]
    assert "\\|" in row, "pipes inside a config label must be escaped"
    # Header defines 6 columns; the data row must have the same cell count.
    header = [l for l in md.splitlines() if l.startswith("| Family")][0]
    assert row.count("|") - row.count("\\|") == header.count("|")


# ---------------------------------------------------------------------------
# Caption wrapping (figure footnotes grow as findings are added)
# ---------------------------------------------------------------------------

def test_figure_caption_wraps_long_text_within_figure_width():
    """Regression: the slides caption grew past the figure width once the seed-band note
    was appended and a centred single line overflowed BOTH edges."""
    from src.compare import _figure_caption
    fig = plt.figure(figsize=(11, 7.5))
    long_text = ("CI whiskers omitted for legibility — 95% session cluster bootstrap, n=11 "
                 "test sessions — intervals are wide; full CIs in results_table.csv / the "
                 "report figure. Vertical bars = PR-AUC range over 5 noise draws, drawn at "
                 "each sigma's mean effective MI. MI-AUC below 0.5 means the attack inverts, "
                 "not extra privacy; effective leakage = max(AUC, 1-AUC).")
    n_lines = _figure_caption(fig, long_text)
    assert n_lines > 1, "a caption this long must wrap"

    # Every rendered line must fit inside the figure with margin to spare.
    fig.canvas.draw()
    texts = [t for t in fig.texts if "CI whiskers" in t.get_text()]
    bb = texts[0].get_window_extent(fig.canvas.get_renderer())
    assert bb.width / fig.dpi <= 11.0, "caption must not exceed the figure width"
    plt.close(fig)


def test_figure_caption_short_text_stays_one_line():
    from src.compare import _figure_caption
    fig = plt.figure(figsize=(11, 7.5))
    assert _figure_caption(fig, "Short caption.") == 1
    plt.close(fig)


def test_markdown_table_footnotes_the_full_note_for_flagged_families():
    """The table must not just say 'does not reach bar': the chosen row is the most
    private config, which can be the family's worst on utility. The within-family Pareto
    alternative has to be visible in the rendered table, not only in the CSV."""
    df = _table([("Input", 0.01, np.nan, 0.200, 0.575),
                 ("Input", 2.0, np.nan, 0.542, 0.585)])
    ops = choose_operating_points(df, mi_bar=DEFAULT_MI_BAR)
    md = markdown_table(ops, mi_bar=DEFAULT_MI_BAR)
    assert "within-family Pareto point" in md, "the alternative must appear in the table"
    assert "0.542" in md and "0.585" in md, "its PR-AUC and eff-MI must be shown"
