"""
Tests for the two decision figures (SUMMARY_FIGURE_TASK.md, UTILITY_RETENTION_TASK.md).

The retention curve is the piece with real logic in it — "best utility subject to leakage
≤ t" is easy to get subtly wrong (extrapolating a family left of its first qualifying
config, or letting the CI standard admit something the point standard would not), so it is
tested against a synthetic table with hand-chosen leakage values rather than only through
"the figure rendered".
"""
import matplotlib
matplotlib.use("Agg")  # headless: no display in CI

import numpy as np
import pandas as pd
import pytest

from src.anchors import DPSGD_METHOD, resolve_anchors
from src.summary_figures import (
    RETENTION_BARS, _keep_inside_axes, anchor_config_label, first_demonstrably_safe_bar,
    plot_headline, plot_sweeps, plot_utility_retention, retention_at, retention_steps,
    retention_table, write_retention_table,
)


def _row(method, sigma, pr, eff_mi, eff_hi, clip=np.nan):
    """One results-table row, carrying only the columns these figures read.

    eff_mi and eff_hi are set INDEPENDENTLY so a config can be estimated-safe while its
    95% upper bound is not — the exact case the two-standard curve exists to show.
    """
    return {"Method": method, "σ / noise_mult": sigma, "clip_norm": clip,
            "PR-AUC": pr, "MI-AUC_eff": eff_mi, "MI-AUC_eff_hi": eff_hi}


@pytest.fixture
def results_df():
    """
    Synthetic table with a deliberately awkward shape per family:

      Input   — its MOST private config is also its WORST on utility (σ=0.01), so a curve
                that just took "the most private qualifying config" would fall as t rises.
      Output  — a config (σ=0.02) that is dominated within its own family: leakier AND
                worse than σ=0.005, so it must never define a step.
      DP-SGD  — every config's upper bound sits past the right end of the sweep.
    """
    rows = [
        _row("Baseline", None, 0.60, 0.70, 0.78),

        _row("Input", 0.01, 0.20, 0.56, 0.66),
        _row("Input", 0.5, 0.45, 0.60, 0.69),
        _row("Input", 2.0, 0.50, 0.62, 0.68),

        _row("Output", 0.005, 0.55, 0.52, 0.63),
        _row("Output", 0.02, 0.40, 0.58, 0.67),   # dominated within the family
        _row("Output", 0.05, 0.58, 0.64, 0.71),

        _row("Output (last-layer)", 0.05, 0.59, 0.66, 0.75),

        _row(DPSGD_METHOD, 1.0, 0.22, 0.57, 0.80, clip=0.5),
        _row(DPSGD_METHOD, 2.0, 0.19, 0.59, 0.82, clip=0.5),

        _row("Personalized", "v1:0.10 | v2:0.09", 0.21, 0.68, 0.79),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def ops_df():
    """An operating-points table shaped like the real one, incl. the alt_* columns."""
    return pd.DataFrame([
        # Missed the bar, gain NOT established, has a better-utility Pareto alt → use alt.
        {"method": "Input", "config": "σ=0.01", "sigma": 0.01, "clip_norm": np.nan,
         "eff_mi": 0.56, "pr_auc": 0.20, "meets_bar": False, "eff_mi_significant": False,
         "alt_config": "σ=2", "alt_sigma": 2.0, "alt_clip_norm": np.nan,
         "alt_eff_mi": 0.62, "alt_pr_auc": 0.50},
        # Missed the bar but the privacy gain IS established → keep the primary row.
        {"method": "Output (last-layer)", "config": "σ=0.05", "sigma": 0.05,
         "clip_norm": np.nan, "eff_mi": 0.66, "pr_auc": 0.59, "meets_bar": False,
         "eff_mi_significant": True, "alt_config": "σ=0.005", "alt_sigma": 0.005,
         "alt_clip_norm": np.nan, "alt_eff_mi": 0.74, "alt_pr_auc": 0.60},
        # Met the bar → primary, regardless of significance.
        {"method": "Output", "config": "σ=0.005", "sigma": 0.005, "clip_norm": np.nan,
         "eff_mi": 0.52, "pr_auc": 0.55, "meets_bar": True, "eff_mi_significant": False,
         "alt_config": np.nan, "alt_sigma": np.nan, "alt_clip_norm": np.nan,
         "alt_eff_mi": np.nan, "alt_pr_auc": np.nan},
    ])


# ---------------------------------------------------------------------------
# Retention: the criterion "keep utility for as long as it is safe"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("standard", ["point", "ci"])
@pytest.mark.parametrize("method", ["Input", "Output", DPSGD_METHOD])
def test_retention_is_monotone_non_decreasing_in_t(results_df, method, standard):
    """Relaxing the privacy bar can only ever ADMIT configs, never remove them."""
    bars = np.arange(0.50, 0.85, 0.01)
    got = [retention_at(results_df, method, b, standard=standard)[0] for b in bars]
    seen = [v for v in got if not np.isnan(v)]
    assert seen == sorted(seen), f"{method}/{standard} retention fell as t rose: {got}"
    # And once a family qualifies it never stops qualifying at a looser bar.
    qualified = [not np.isnan(v) for v in got]
    assert qualified == sorted(qualified), "a family stopped qualifying as t rose"


def test_retention_is_empty_below_first_config(results_df):
    """
    Left of its first qualifying config a family has NO curve — not a zero.

    "No config of this family is this private" and "this family retains 0% utility" are
    different claims, and drawing the second when the first is true would invent a
    catastrophic-utility point the study never measured.
    """
    first = retention_steps(results_df, "Output", standard="point")["threshold"].min()
    assert first == pytest.approx(0.52)
    pr, config = retention_at(results_df, "Output", first - 0.001, standard="point")
    assert np.isnan(pr) and config is None
    # Exactly AT the bar the config qualifies (the constraint is leakage ≤ t).
    assert retention_at(results_df, "Output", first, standard="point")[0] == pytest.approx(0.55)


@pytest.mark.parametrize("method", ["Input", "Output", "Output (last-layer)", DPSGD_METHOD])
def test_ci_standard_is_never_looser_than_point_standard(results_df, method):
    """
    Demanding the whole 95% interval clear the bar can only ever be stricter.

    Since MI-AUC_eff ≤ MI-AUC_eff_hi by construction, any config qualifying under the CI
    standard also qualifies under the point standard — so the dashed curve must sit at or
    below the solid one everywhere, never above it.
    """
    for bar in np.arange(0.50, 0.85, 0.01):
        point = retention_at(results_df, method, bar, standard="point")[0]
        ci = retention_at(results_df, method, bar, standard="ci")[0]
        if np.isnan(ci):
            continue
        assert not np.isnan(point), f"{method}: CI standard qualified at t={bar} but point did not"
        assert ci <= point + 1e-12, f"{method}: CI retention {ci} exceeded point {point} at t={bar}"


def test_retention_steps_skips_configs_dominated_within_the_family(results_df):
    """Output σ=0.02 is leakier AND worse than σ=0.005, so it never defines a step."""
    steps = retention_steps(results_df, "Output", standard="point")
    assert list(steps["config"]) == ["σ=0.005", "σ=0.05"]
    assert list(steps["pr_auc"]) == pytest.approx([0.55, 0.58])


def test_retention_picks_best_utility_not_most_private(results_df):
    """
    At a bar admitting several Input configs, the curve takes the most ACCURATE one.

    Input's most private config (σ=0.01) is also its worst on utility; a curve that took
    "most private qualifying" would report 33% where 83% is available.
    """
    pr, config = retention_at(results_df, "Input", 0.62, standard="point")
    assert config == "σ=2" and pr == pytest.approx(0.50)


def test_retention_table_has_a_column_block_per_family_and_nan_where_none_qualify(results_df):
    table = retention_table(results_df, bars=RETENTION_BARS)
    assert list(table["privacy_bar_eff_mi"]) == list(RETENTION_BARS)
    for family in ("Input", "Output", "Output-LL", "DP-SGD", "Personalized"):
        for standard in ("point", "ci"):
            assert f"{family}_config_{standard}" in table.columns
            assert f"{family}_retained_{standard}_pct" in table.columns
    # At the strictest bar nothing in the fixture is that private, under either standard.
    strict = table.iloc[0]
    assert np.isnan(strict["Output_retained_point_pct"])
    assert np.isnan(strict["Output_retained_ci_pct"])
    # Retention is a PERCENTAGE of the 0.60 baseline, so Output's 0.55 reads as ~91.7%.
    loose = table[table["privacy_bar_eff_mi"] == 0.55].iloc[0]
    assert loose["Output_retained_point_pct"] == pytest.approx(100 * 0.55 / 0.60)


def test_retention_table_is_written(results_df, tmp_path):
    write_retention_table(results_df, str(tmp_path))
    written = pd.read_csv(tmp_path / "utility_retention.csv")
    assert len(written) == len(RETENTION_BARS)


# ---------------------------------------------------------------------------
# Anchors: which config represents each family
# ---------------------------------------------------------------------------

def test_anchor_falls_back_to_pareto_point_when_gain_is_not_established(ops_df):
    """
    A family that missed the bar with an unproven privacy gain is shown at its own
    best-utility Pareto point, not at the near-worthless most-private config.
    """
    a = resolve_anchors(ops_df).set_index("method")
    assert a.loc["Input", "config"] == "σ=2"
    assert a.loc["Input", "pr_auc"] == pytest.approx(0.50)
    assert bool(a.loc["Input", "is_alt"])


def test_anchor_keeps_the_primary_config_when_the_gain_is_established(ops_df):
    """Output-LL bought real leakage reduction, so its own row stands."""
    a = resolve_anchors(ops_df).set_index("method")
    assert a.loc["Output (last-layer)", "config"] == "σ=0.05"
    assert not bool(a.loc["Output (last-layer)", "is_alt"])


def test_anchor_keeps_the_primary_config_when_the_bar_is_met(ops_df):
    a = resolve_anchors(ops_df).set_index("method")
    assert a.loc["Output", "config"] == "σ=0.005"
    assert not bool(a.loc["Output", "is_alt"])


def test_every_anchor_carries_a_verdict(ops_df):
    assert all(str(v).strip() for v in resolve_anchors(ops_df)["verdict"])


def test_resolve_anchors_tolerates_a_missing_table():
    """A checkout that never ran the decision view still renders figures, sans anchors."""
    assert resolve_anchors(None) is None
    assert resolve_anchors(pd.DataFrame()) is None


def test_anchor_config_label_is_short_enough_to_sit_beside_a_point():
    """On-figure text is names and configs only, so the config text must actually fit."""
    # Personalized's config is a 45-char per-vehicle σ map; it names its shape instead.
    assert anchor_config_label(
        {"method": "Personalized", "sigma": "v1:0.104 | v2:0.097 | v3:0.056"}) == "per-vehicle σ"
    # Logspace σ print as '0.009180942678743854' raw; 2 significant figures on the figure.
    assert anchor_config_label({"method": "Output", "sigma": 0.009180942678743854}) == "σ=0.0092"
    assert anchor_config_label({"method": "Input", "sigma": 2.0}) == "σ=2"
    # A DP-SGD run is a (clip, noise) cell — 'nm=1' alone loses the clip.
    assert anchor_config_label(
        {"method": DPSGD_METHOD, "sigma": 1.0, "clip_norm": 0.5}) == "C=0.5, nm=1"


# ---------------------------------------------------------------------------
# Layout invariants (the task's "no clipped labels")
# ---------------------------------------------------------------------------

def test_keep_inside_axes_pulls_an_overhanging_label_back(results_df):
    """The guarantee behind 'no clipped labels': an off-axes label is moved back on."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Deliberately shoved far off the right edge.
    ann = ax.annotate("a very long label indeed", (0.98, 0.5),
                      textcoords="offset points", xytext=(400, 0), ha="left")
    _keep_inside_axes(fig, ax, [ann])
    ext, box = ann.get_window_extent(), ax.get_window_extent()
    assert ext.x1 <= box.x1 + 1e-6 and ext.x0 >= box.x0 - 1e-6
    plt.close(fig)


def test_scale_annotations_shrinks_labels_for_the_report_copy():
    """
    The PDF is the same figure re-saved narrower; points don't shrink, so labels must.

    Also checked: repeated calls do NOT compound, because the export path re-runs the
    layout once at report width and again on restore.
    """
    import matplotlib.pyplot as plt
    from src.summary_figures import scale_annotations

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ann = ax.annotate("x", (0.5, 0.5), textcoords="offset points", xytext=(10, 20),
                      fontsize=8.5)
    base = [(ann.get_fontsize(), ann.get_position())]

    k = scale_annotations([ann], base, 6.5, 10.5)
    assert k == pytest.approx(6.5 / 10.5)
    assert ann.get_fontsize() == pytest.approx(8.5 * k)
    assert ann.get_position() == pytest.approx((10 * k, 20 * k))

    scale_annotations([ann], base, 6.5, 10.5)  # idempotent, not compounding
    assert ann.get_fontsize() == pytest.approx(8.5 * k)

    scale_annotations([ann], base, 10.5, 10.5)  # back to design size
    assert ann.get_fontsize() == pytest.approx(8.5)
    # Never enlarged past the design size, however wide the canvas.
    assert scale_annotations([ann], base, 20.0, 10.5) == 1.0
    plt.close(fig)


def test_style_figure_rewraps_instead_of_stacking_a_second_title():
    """
    The PDF is the same figure re-laid-out narrower, so style_figure runs more than once.

    It must REWRAP the existing artists rather than draw a second title on top of the
    first — and the narrower canvas must actually cost the text more lines, which is what
    reserves the vertical space the axes then shrink into.
    """
    import matplotlib.pyplot as plt
    from src import plotstyle

    fig, _ = plt.subplots(figsize=(10.5, 6))
    title = ("Adding noise to model outputs drives the attack to chance while keeping "
             "91% of detection")
    plotstyle.style_figure(fig, title, "a subtitle", "a caption")
    n_before = len(fig.texts)
    wide = fig.__dict__["_plotstyle_text"]["title"].get_text()

    fig.set_size_inches(6.5, 6.5 * 6 / 10.5)
    plotstyle.style_figure(fig, title, "a subtitle", "a caption")
    narrow = fig.__dict__["_plotstyle_text"]["title"].get_text()

    assert len(fig.texts) == n_before, "second call stacked new artists instead of reusing"
    assert narrow.count("\n") > wide.count("\n"), "narrow canvas must wrap to more lines"
    # Wrapped, never truncated: the sentence still says what it said.
    assert narrow.replace("\n", " ") == title
    plt.close(fig)


# ---------------------------------------------------------------------------
# Acceptance: the figures render
# ---------------------------------------------------------------------------

def test_all_presentation_figures_render(results_df, ops_df, tmp_path):
    anchors = resolve_anchors(ops_df)
    d = str(tmp_path)
    plot_headline(results_df, d, anchors=anchors)
    plot_sweeps(results_df, d)
    plot_utility_retention(results_df, d, anchors=anchors)
    for name in ("fig_headline.png", "fig_sweeps.png", "utility_retention.png"):
        assert (tmp_path / name).exists() and (tmp_path / name).stat().st_size > 0
    # Each gets a vector copy for the report, in the sibling figures_pdf/.
    pdf_dir = tmp_path.parent / "figures_pdf"
    for name in ("fig_headline.pdf", "fig_sweeps.pdf", "utility_retention.pdf"):
        assert (pdf_dir / name).exists()


def test_presentation_figures_render_without_anchors(results_df, tmp_path):
    """No operating_points.csv → figures still render, just without the headline points."""
    d = str(tmp_path)
    plot_headline(results_df, d, anchors=None, out_dir=d)
    plot_sweeps(results_df, d)
    plot_utility_retention(results_df, d, anchors=None, out_dir=d)
    assert (tmp_path / "fig_headline.png").exists()
    assert (tmp_path / "fig_sweeps.png").exists()
    assert (tmp_path / "utility_retention.png").exists()


def test_sweeps_skips_families_that_have_no_sigma_axis(results_df, tmp_path):
    """
    DP-SGD and Personalized must not get a σ column invented for them.

    DP-SGD's runs are a (clip × noise) grid and Personalized is a per-vehicle σ map, so
    neither has a single σ to put on an axis — plotting them here would fabricate one.
    """
    from src.summary_figures import SWEEP_FAMILIES
    assert DPSGD_METHOD not in SWEEP_FAMILIES
    assert "Personalized" not in SWEEP_FAMILIES

    # With no σ-sweep family present at all, the figure declines to render rather than
    # drawing empty panels.
    only_dpsgd = results_df[results_df["Method"].isin(["Baseline", DPSGD_METHOD])]
    plot_sweeps(only_dpsgd, str(tmp_path))
    assert not (tmp_path / "fig_sweeps.png").exists()


def test_first_demonstrably_safe_bar_is_the_strictest_ci_threshold(results_df):
    """
    The retention caption quotes this number in place of the dashed CI curves it replaced,
    so it must be the strictest bar at which ANY family qualifies under the CI standard.
    """
    from src.summary_figures import _families
    bar = first_demonstrably_safe_bar(results_df)
    per_family = [retention_steps(results_df, m, standard="ci")["threshold"].min()
                  for m in _families(results_df)
                  if not retention_steps(results_df, m, standard="ci").empty]
    assert bar == pytest.approx(min(per_family))
    # And it is genuinely stricter than what the point estimate admits — that gap is the
    # whole reason the sentence exists.
    point_first = min(retention_steps(results_df, m, standard="point")["threshold"].min()
                      for m in _families(results_df)
                      if not retention_steps(results_df, m, standard="point").empty)
    assert bar > point_first
