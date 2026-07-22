"""
Tests for the report-tier CARLA figures.

These drive the plotting code with a synthetic results table in the real schema, so they
catch wiring/label/geometry regressions without needing the 3M-window CARLA dataset.
"""
import matplotlib
matplotlib.use("Agg")  # headless: no display in CI

import numpy as np
import pandas as pd
import pytest

from src.compare import (
    _RESULT_HEADERS, DPSGD_METHOD,
    _sigma_label, _point_label, _split_by_sigma,
    _plot_epsilon, _plot_mi_report, _plot_dpsgd_grid,
)
from src.anchors import SHORT_NAME
from src.reeval import REEVAL_OUTPUT_SIGMAS, EXTENDED_INPUT_SIGMAS


def _row(method, sigma, clip, eps, pr, mi):
    """One results row as a header-keyed dict (robust to schema column additions).

    `mi` is the RAW MI-AUC; the effective columns are derived as max(raw, 1-raw) so the
    fixture exercises the inverted-attack (raw < 0.5) path when mi < 0.5.
    """
    eff = max(mi, 1.0 - mi)
    return {
        "Method": method, "σ / noise_mult": sigma, "clip_norm": clip,
        "ε": eps, "ε (final epoch)": np.nan,
        "PR-AUC": pr, "PR-AUC_lo": pr - 0.11, "PR-AUC_hi": pr + 0.07, "prevalence": 0.05,
        "AUROC": 0.9, "AUROC_lo": 0.85, "AUROC_hi": 0.92,
        "F1": 0.8, "FPR@95TPR": 0.55, "MSE(normal)": 0.013,
        "recall_fuzzy": 0.95, "recall_plateau": 0.10,
        "AUROC_fuzzy": 0.99, "AUROC_plateau": 0.55,
        "MI-AUC": mi, "MI-AUC_lo": mi - 0.13, "MI-AUC_hi": mi + 0.06,
        "MI-AUC_eff": eff, "MI-AUC_eff_lo": eff - 0.08, "MI-AUC_eff_hi": eff + 0.08,
        # Cost columns (TIMING_TASK.md part B). Output families are DERIVED from the
        # baseline checkpoint, so they carry derive_seconds and no training time; every
        # other family is trained. Figures ignore these, but the fixture is meant to be
        # shaped exactly like the real table.
        **({"train_seconds": np.nan, "epochs_ran": np.nan, "sec_per_epoch": np.nan,
            "derive_seconds": 0.004}
           if str(method).startswith("Output")
           else {"train_seconds": 300.0, "epochs_ran": 20, "sec_per_epoch": 15.0,
                 "derive_seconds": np.nan}),
        "device": "cpu", "hostname": "test-host",
    }


@pytest.fixture
def results_df():
    """A synthetic results table shaped exactly like the real one (dense output sweep).

    The Output family's raw MI-AUC crosses below 0.5 at high σ, so the fixture also
    exercises the inverted-attack (hollow marker) rendering path.
    """
    rows = [_row("Baseline", None, np.nan, np.inf, 0.612, 0.704)]
    for s, pr, mi in zip([0.01, 0.05, 0.1, 0.3, "per-feature"],
                         [0.20, 0.203, 0.21, 0.315, 0.198],
                         [0.575, 0.660, 0.681, 0.677, 0.585]):
        rows.append(_row("Input", s, np.nan, 1e4, pr, mi))
    for i, s in enumerate(REEVAL_OUTPUT_SIGMAS):
        # raw MI-AUC dips below 0.5 at the top of the sweep (inverted attack).
        rows.append(_row("Output", s, np.nan, 1.6e11 / (s / 1e-4), 0.612 * (1 - i / 13), 0.704 - 0.03 * i))
        rows.append(_row("Output (last-layer)", s, np.nan, 9e9 / (s / 1e-4), 0.612 - 0.001 * i, 0.7047 + 0.0033 * i))
    for clip in [0.5, 1.0, 5.0]:
        for nm in [1.0, 2.0]:
            rows.append(_row(DPSGD_METHOD, nm, clip, 0.05 * clip * nm, 0.22 - 0.02 * nm, 0.57 + 0.01 * clip))
    rows.append(_row("Personalized", "v1:0.104 | v2:0.097", np.nan, np.nan, 0.215, 0.671))
    return pd.DataFrame(rows)[list(_RESULT_HEADERS)]


# ---------------------------------------------------------------------------
# Sweep definitions (Tier 1a / Tier 2)
# ---------------------------------------------------------------------------

def test_reeval_output_sweep_is_dense_and_log_spaced():
    """Tier 1a: ~12 log-spaced σ from 1e-4 to 5e-2 (evaluation-only, so it is free)."""
    s = np.array(REEVAL_OUTPUT_SIGMAS)
    assert len(s) == 12
    assert s[0] == pytest.approx(1e-4) and s[-1] == pytest.approx(5e-2)
    assert (np.diff(s) > 0).all(), "σ must be ascending"
    # Even spacing in log space ⇒ constant ratio between neighbours.
    ratios = s[1:] / s[:-1]
    assert np.allclose(ratios, ratios[0]), "σ must be log-spaced"


def test_reeval_output_sweep_denser_than_training_sweep():
    from src.compare import OUTPUT_SIGMAS
    assert len(REEVAL_OUTPUT_SIGMAS) > len(OUTPUT_SIGMAS)
    # compare.py's training path must be left untouched by the densification.
    assert OUTPUT_SIGMAS == [0.0001, 0.001, 0.005, 0.01]


def test_extended_input_sigmas_reach_the_degraded_regime():
    """Tier 2 exists to push past σ=0.3, where the trained sweep stops."""
    assert min(EXTENDED_INPUT_SIGMAS) > 0.3


# ---------------------------------------------------------------------------
# Labelling helpers
# ---------------------------------------------------------------------------

def test_sigma_label_is_compact_for_logspace_values():
    """Raw logspace σ print as '0.00521827'; labels must stay short."""
    assert all(len(_sigma_label(s)) <= 6 for s in REEVAL_OUTPUT_SIGMAS)
    assert _sigma_label(1e-4) == "1.0e-4"
    assert _sigma_label(0.05) == "0.05"


def test_sigma_label_passes_through_non_numeric():
    assert _sigma_label("per-feature") == "per-feature"


def test_point_label_identifies_dpsgd_cell_not_bare_sigma():
    """A DP-SGD point is a (clip, noise) cell — 'nm=1' alone loses the clip."""
    row = pd.Series({"Method": DPSGD_METHOD, "σ / noise_mult": 1.0, "clip_norm": 0.5})
    assert _point_label(row) == "C=0.5/nm=1"


def test_split_by_sigma_orders_by_sigma_and_separates_non_numeric(results_df):
    sub = results_df[results_df["Method"] == "Input"]
    numeric, other = _split_by_sigma(sub)
    # Ordered by σ (the sweep trajectory), NOT by the metric.
    assert list(numeric["σ / noise_mult"]) == [0.01, 0.05, 0.1, 0.3]
    assert list(other["σ / noise_mult"]) == ["per-feature"]


# ---------------------------------------------------------------------------
# Figures render (acceptance)
# ---------------------------------------------------------------------------

def test_all_report_tier_figures_render(results_df, tmp_path):
    d = str(tmp_path)
    baseline = results_df[results_df["Method"] == "Baseline"].iloc[0]
    _plot_epsilon(results_df, d)
    _plot_mi_report(results_df, d, baseline, 0.05)
    _plot_dpsgd_grid(results_df, d, 0.05)
    for name in ("privacy_utility_epsilon.png", "privacy_utility_mi_report.png",
                 "dpsgd_grid.png"):
        assert (tmp_path / name).exists(), f"{name} not written"
        assert (tmp_path / name).stat().st_size > 0


def test_inverted_mask_flags_below_half_raw_mi(results_df):
    """Rows with raw MI-AUC < 0.5 (Output at high σ) are the inverted-attack case.

    No longer drives a hollow marker (markers encode nothing but "a config" now) — it is
    the data property the effective-MI captions describe, so it still has to be right.
    """
    from src.compare import _inverted_mask
    out = results_df[results_df["Method"] == "Output"]
    assert _inverted_mask(out).any(), "fixture should contain inverted Output points"
    # None of the Input rows invert (all raw MI-AUC > 0.5).
    assert not _inverted_mask(results_df[results_df["Method"] == "Input"]).any()


def test_dpsgd_grid_skips_cleanly_without_dpsgd_rows(results_df, tmp_path):
    """Partial model sets (no Opacus) must not crash the figure pass."""
    no_dpsgd = results_df[results_df["Method"] != DPSGD_METHOD]
    _plot_dpsgd_grid(no_dpsgd, str(tmp_path), 0.05)
    assert not (tmp_path / "dpsgd_grid.png").exists()


# ---------------------------------------------------------------------------
# The structural rule: lines only where y is a function of x
# ---------------------------------------------------------------------------

def _capture_figure(monkeypatch, draw):
    """
    Run a plot function and hand back its Figure before the function closes it.

    Patches save_figure on src.compare specifically, because that is where it is defined —
    src.summary_figures imports it lazily inside each plot function (to avoid an import
    cycle), so it picks up the patched version at call time too.
    """
    import matplotlib.pyplot as plt
    import src.compare as compare
    captured = {}
    real_save = compare.save_figure

    def spy(fig, *args, **kwargs):
        captured["fig"] = fig
        return real_save(fig, *args, **kwargs)

    monkeypatch.setattr(compare, "save_figure", spy)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    draw()
    return captured["fig"]


@pytest.fixture
def ops_df():
    """Minimal operating-points table — one anchor per family for the headline figure."""
    return pd.DataFrame([
        {"method": "Input", "config": "σ=2", "sigma": 2.0, "clip_norm": np.nan,
         "eff_mi": 0.585, "pr_auc": 0.542, "meets_bar": False, "eff_mi_significant": False},
        {"method": "Output", "config": "σ=0.009", "sigma": 0.009, "clip_norm": np.nan,
         "eff_mi": 0.507, "pr_auc": 0.556, "meets_bar": True, "eff_mi_significant": False},
        {"method": DPSGD_METHOD, "config": "C=0.5, nm=1", "sigma": 1.0, "clip_norm": 0.5,
         "eff_mi": 0.566, "pr_auc": 0.218, "meets_bar": False, "eff_mi_significant": False},
    ])


def _connecting_lines(ax):
    """
    Labels of any multi-point drawn line on `ax`, excluding reference rules.

    A 2-point line is an axhline/axvline (baseline, chance); anything with more vertices
    and a visible linestyle is joining data points together.
    """
    out = []
    for line in ax.lines:
        if line.get_linestyle() in ("none", "None", " ", ""):
            continue
        if len(line.get_xdata()) <= 2:  # axhline / axvline reference rule
            continue
        out.append(line.get_label())
    return out


def test_tradeoff_figure_connects_no_points(results_df, tmp_path, monkeypatch):
    """
    The whole point of the redesign: in metric-vs-metric space, points are never joined.

    Both axes here are metrics, so a line between two configs traces the hidden σ. Where
    leakage is non-monotonic in σ — which the Output fixture reproduces — that path folds
    back and reads as two branches of a curve that does not exist. Only the Pareto frontier
    staircase may be a line, because it is a boundary, not a trajectory.

    Checked across EVERY facet: the figure is now small multiples, so testing one panel
    would leave three families free to draw whatever they liked.
    """
    import src.compare as C
    baseline = results_df[results_df["Method"] == "Baseline"].iloc[0]
    fig = _capture_figure(monkeypatch, lambda: C._plot_mi_report(
        results_df, str(tmp_path), baseline, 0.05))
    offenders = [l for ax in fig.axes for l in _connecting_lines(ax) if "Pareto" not in l]
    assert not offenders, f"points joined in metric-vs-metric space: {offenders}"


def test_mi_report_facets_one_panel_per_family_and_draws_the_frontier_in_each(
        results_df, tmp_path, monkeypatch):
    """
    Small multiples, with the SAME frontier repeated — that pairing is the whole redesign.

    A shared frontier is what lets "does this family reach the boundary?" be read per
    panel; drawing each family its own frontier would make every panel touch its own line
    and answer the question 'yes' four times.
    """
    import src.compare as C
    baseline = results_df[results_df["Method"] == "Baseline"].iloc[0]
    fig = _capture_figure(monkeypatch, lambda: C._plot_mi_report(
        results_df, str(tmp_path), baseline, 0.05))
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert len(panels) == len(C.MI_REPORT_FACETS)
    assert [ax.get_title() for ax in panels] == [
        SHORT_NAME.get(m, m) for m in C.MI_REPORT_FACETS]

    frontiers = [[l for l in _connecting_lines(ax) if "Pareto" in l] for ax in panels]
    assert all(len(f) == 1 for f in frontiers), "every facet needs the frontier"
    # Identical geometry in every panel = one boundary shown four times, not four boundaries.
    stairs = [next(l for l in ax.lines if "Pareto" in str(l.get_label())) for ax in panels]
    first = stairs[0].get_xydata()
    assert all(np.array_equal(s.get_xydata(), first) for s in stairs[1:])


def test_mi_report_gives_personalized_no_panel(results_df, tmp_path, monkeypatch):
    import src.compare as C
    baseline = results_df[results_df["Method"] == "Baseline"].iloc[0]
    fig = _capture_figure(monkeypatch, lambda: C._plot_mi_report(
        results_df, str(tmp_path), baseline, 0.05))
    assert "Personalized" not in [ax.get_title() for ax in fig.axes if ax.get_visible()]
    # …and the row itself survives in the table, which is the other half of the decision.
    assert "Personalized" in set(results_df["Method"])


def test_mi_report_frontier_ignores_the_excluded_family(tmp_path, monkeypatch):
    """
    A family that is not DRAWN must not define the boundary the drawn ones are judged by.

    The shared fixture cannot show this — its Personalized row is dominated, so including
    it moves nothing. So this builds the case that matters: a Personalized point that is
    the most private thing in the table and therefore WOULD own the frontier's left end.
    If the exclusion were applied only to the panels, the staircase would step up at
    x=0.50 and every real family would be measured against a boundary set by a config the
    reader cannot see.
    """
    import src.compare as C

    rows = [_row("Baseline", None, np.nan, np.inf, 0.612, 0.704),
            _row("Input", 0.1, np.nan, 1e4, 0.21, 0.68),
            _row("Output", 0.005, np.nan, 1e6, 0.55, 0.60),
            # Most private in the table AND not dominated → it would own the frontier.
            _row("Personalized", "v1:0.1", np.nan, np.nan, 0.30, 0.50)]
    df = pd.DataFrame(rows)[list(_RESULT_HEADERS)]
    baseline = df[df["Method"] == "Baseline"].iloc[0]

    assert "Personalized" in set(C.pareto_frontier(df)["Method"]), \
        "fixture must place Personalized ON the all-inclusive frontier, or this proves nothing"

    fig = _capture_figure(monkeypatch,
                          lambda: C._plot_mi_report(df, str(tmp_path), baseline, 0.05))
    panel = next(ax for ax in fig.axes if ax.get_visible())
    drawn_stair = next(ln for ln in panel.lines
                       if "Pareto" in str(ln.get_label())).get_xydata()

    y0 = panel.get_ylim()[0]
    expected = np.array(C._pareto_staircase(
        C.pareto_frontier(df[~df["Method"].isin(C.FIGURE_EXCLUDED)]), *C.MI_XLIM, y0)).T
    with_excluded = np.array(C._pareto_staircase(
        C.pareto_frontier(df), *C.MI_XLIM, y0)).T

    assert np.allclose(drawn_stair, expected)
    assert not np.array_equal(expected, with_excluded), \
        "the two frontiers must actually differ, or the assertion above is vacuous"


def test_headline_figure_connects_no_points(results_df, ops_df, tmp_path, monkeypatch):
    """Same rule on the presentation-tier tradeoff figure, which has no frontier either."""
    import src.summary_figures as S
    from src.anchors import resolve_anchors
    fig = _capture_figure(monkeypatch, lambda: S.plot_headline(
        results_df, str(tmp_path), anchors=resolve_anchors(ops_df)))
    assert not _connecting_lines(fig.axes[0])


def test_sweeps_figure_does_connect_its_points(results_df, tmp_path, monkeypatch):
    """
    The converse, so the rule above cannot be satisfied by simply never drawing a line.

    On fig_sweeps x IS σ, so both metrics are genuine functions of it and the sweep must be
    drawn as a real line chart — that is the whole reason this figure exists.
    """
    import src.summary_figures as S
    fig = _capture_figure(monkeypatch, lambda: S.plot_sweeps(results_df, str(tmp_path)))
    assert all(_connecting_lines(ax) for ax in fig.axes), \
        "every sweep panel should join its points — x is σ here"


# ---------------------------------------------------------------------------
# The prose strip (FIGURE_CLEANUP_TASKS.md): no sentences on any figure
# ---------------------------------------------------------------------------

def test_style_figure_refuses_a_caption_that_has_grown_back_into_prose():
    """
    The strip is enforced, not just performed.

    Every figure here once carried a takeaway-sentence title, a subtitle sentence and a
    caption paragraph, and they came back the last time by being appended one clause at a
    time. A hard ceiling is what makes that regression fail loudly instead of silently
    re-growing the text block.
    """
    import matplotlib.pyplot as plt
    from src import plotstyle

    fig = plt.figure(figsize=(8, 5))
    short = "Personalized excluded — single-client simulation, no formal ε"
    assert len(short) <= plotstyle.CAPTION_CLAUSE_MAX
    plotstyle.style_figure(fig, "A noun phrase", caption=short)   # allowed

    prose = ("Curves use each config's POINT estimate of leakage. Requiring the whole 95% "
             "interval to clear the bar is far stricter: no family qualifies below t=0.70 "
             "at n=11 test sessions.")
    with pytest.raises(ValueError, match="Figure_Explainer"):
        plotstyle.style_figure(fig, "A noun phrase", caption=prose)
    plt.close(fig)


def test_reference_lines_are_tagged_with_one_lowercase_word():
    """
    The locked decision: chance/baseline rules label themselves, identically everywhere.

    With the subtitles gone there is nothing left on a stripped figure to explain a bare
    rule at 0.5, so the tag defaults ON — a figure has to opt OUT (shared axes), never in.
    """
    import matplotlib.pyplot as plt
    from src import plotstyle

    assert plotstyle.CHANCE_LABEL == "chance"
    assert plotstyle.BASELINE_LABEL == "baseline"

    fig, ax = plt.subplots()
    plotstyle.chance_line(ax, 0.5)
    plotstyle.baseline_line(ax, 0.6)
    assert sorted(t.get_text() for t in ax.texts) == ["baseline", "chance"]

    fig2, ax2 = plt.subplots()
    plotstyle.chance_line(ax2, 0.5, label=None)
    plotstyle.baseline_line(ax2, 0.6, label=None)
    assert not ax2.texts, "opting out must leave no tag at all"
    plt.close(fig)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# The ε range chart (replaces the two-panel PR-AUC-vs-ε curves)
# ---------------------------------------------------------------------------

def test_epsilon_spans_are_per_family_min_max_over_finite_values(results_df):
    from src.compare import _epsilon_spans

    spans = dict((m, (lo, hi)) for m, lo, hi in _epsilon_spans(results_df))
    for method, (lo, hi) in spans.items():
        eps = pd.to_numeric(results_df[results_df["Method"] == method]["ε"],
                            errors="coerce")
        eps = eps[np.isfinite(eps) & (eps > 0)]
        assert lo == pytest.approx(eps.min()) and hi == pytest.approx(eps.max())
    # Personalized has no analytic ε, so it gets no row rather than a fabricated one.
    assert "Personalized" not in spans
    # The baseline is not a defence family and never gets a budget row.
    assert "Baseline" not in spans


def test_epsilon_chart_draws_one_row_per_family_and_the_meaningful_zone(
        results_df, tmp_path, monkeypatch):
    """The chart's whole message is which families fall inside ε ≤ 10 — so both the rows
    and the zone boundary have to actually be on the axes."""
    import src.compare as C

    fig = _capture_figure(monkeypatch,
                          lambda: C._plot_epsilon(results_df, str(tmp_path)))
    ax = fig.axes[0]
    spans = C._epsilon_spans(results_df)
    assert len(ax.get_yticks()) == len(spans)
    assert ax.get_xscale() == "log", "ε spans 13 decades; a linear axis hides all of it"
    zone = [ln for ln in ax.lines
            if len(ln.get_xdata()) == 2
            and ln.get_xdata()[0] == pytest.approx(C.EPSILON_MEANINGFUL_MAX)]
    assert zone, "the ε ≤ 10 boundary must be drawn"


def test_epsilon_labels_stay_short_across_thirteen_decades():
    """0.0496 must read as a number and 1.6e11 as a magnitude — neither as 12 digits."""
    from src.compare import _fmt_epsilon

    assert _fmt_epsilon(0.0496775) == "0.05"
    assert _fmt_epsilon(20.774) == "21"
    assert _fmt_epsilon(157929896246.96) == "1.6e11"
    assert all(len(_fmt_epsilon(v)) <= 6
               for v in (0.0496775, 20.774, 183366.2, 9021158723.58, 157929896246.96))


# ---------------------------------------------------------------------------
# Personalized: excluded from figures, retained in the record
# ---------------------------------------------------------------------------

def test_personalized_is_excluded_from_figures_but_kept_in_the_tables(results_df):
    """
    Both halves of the decision, asserted together, because either alone is a bug.

    Dropping it from the figures without keeping the row would delete a measured
    experiment; keeping the row without dropping it from the figures is the state this
    pass exists to fix. It is the bridge to the FL phase, so the data must survive.
    """
    from src.anchors import FIGURE_EXCLUDED, figure_families

    assert "Personalized" in FIGURE_EXCLUDED
    families = list(results_df["Method"].unique())
    assert "Personalized" in families, "fixture must carry the row"
    assert "Personalized" not in figure_families(families)
    # Nothing else is swept up by the exclusion.
    assert set(families) - set(figure_families(families)) == {"Personalized"}


def test_retention_table_still_reports_personalized(results_df):
    """utility_retention.csv is a table, not a figure — the exclusion must not reach it."""
    from src.summary_figures import retention_table

    table = retention_table(results_df)
    assert any("Personalized" in c for c in table.columns)


def test_retired_kaggle_figures_are_deleted_from_both_trees(tmp_path):
    """
    A cut figure must be REMOVED, not merely left un-regenerated.

    The names survive in old notes and slide decks, so a stale PNG sitting in the output
    directory is exactly how a retired figure reaches a deck — worse than one that is gone.
    """
    import src.compare_kaggle as CK

    figures = tmp_path / "kaggle_figures"
    pdfs = tmp_path / "figures_pdf"
    figures.mkdir()
    pdfs.mkdir()
    assert "kaggle_dotplot_slides.png" in CK._RETIRED_FIGURES

    (figures / "kaggle_dotplot_slides.png").write_bytes(b"stale")
    (pdfs / "kaggle_dotplot_slides.pdf").write_bytes(b"stale")
    keep = figures / "kaggle_dotplot.png"
    keep.write_bytes(b"current")

    CK._remove_retired_figures(str(figures))

    assert not (figures / "kaggle_dotplot_slides.png").exists()
    assert not (pdfs / "kaggle_dotplot_slides.pdf").exists()
    assert keep.exists(), "the surviving dot plot must not be swept up"


def test_removing_retired_figures_is_safe_when_they_were_never_written(tmp_path):
    """A fresh checkout has none of them; the sweep must be a no-op, not a crash."""
    import src.compare_kaggle as CK

    figures = tmp_path / "kaggle_figures"
    figures.mkdir()
    CK._remove_retired_figures(str(figures))
