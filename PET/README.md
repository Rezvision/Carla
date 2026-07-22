# PET_Code

GRU-based intrusion detection for vehicle CAN bus telemetry, with differential
privacy (input / output / gradient perturbation) as a defence. Studies the
privacy–utility tradeoff.

## Layout
- `notebooks/` — current model code (`workflow.ipynb`) + saved `models/model.pth`
- `data/CARLA_processed/` — working dataset, 103 sessions / 3.16M rows (gitignored)
- `src/` — refactored, importable code
- `tests/` — pytest tests

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch opacus pytest pandas pyarrow scikit-learn matplotlib
```

## Goal (this week)
Baseline GRU autoencoder IDS → input, output, and gradient (DP-SGD/Opacus)
perturbation variants over several noise levels → one privacy-vs-utility comparison.

See `CLAUDE.md` for full project context.

## Training cost (the third axis)

Wall-clock cost sits alongside privacy and utility: at comparable (ε, PR-AUC), a family
that needs a full retrain per σ is a different proposition from one derived from an
existing checkpoint. Three complementary sources, in increasing order of trustworthiness:

```bash
# A. Retrospective ESTIMATES from checkpoint mtimes (no compute).
#    Only valid on the machine that trained the models — git does not preserve mtimes.
python -m src.reeval --timing_retrospective        # → notebooks/timing_retrospective.csv

# C. Clean same-machine benchmark: fixed 3 epochs per family, early stopping disabled.
python -m src.reeval --timing_benchmark            # → notebooks/timing_benchmark.csv
                                                   #   figures/timing_by_family.png
                                                   #   notebooks/timing_summary.md (part D)
```

B. Every future sweep is instrumented automatically: `results_table.csv` gains
`train_seconds`, `epochs_ran`, `sec_per_epoch`, `derive_seconds`, `device` and `hostname`.
Models *derived* from a checkpoint (output / output-last-layer) record `derive_seconds`
rather than training time — the near-zero value is the finding, so it is never left blank.
The cost of one full evaluation (dominated by the bootstrap CIs) is measured once per run
and written to `notebooks/timing_run_meta.csv`, kept out of the per-model table because it
describes the eval harness, not any single model.

## Which defence should I pick?

Every report-tier figure draws all 40 configs, which reads a family's shape well and answers
"which one do I use?" badly. Three presentation-tier figures lead the set instead:

```bash
# All three derive from results_table.csv + operating_points.csv — no dataset, no checkpoints.
python -m src.summary_figures --out_dir notebooks   # → figures/fig_headline.png
                                                    #   figures/fig_sweeps.png
                                                    #   figures/utility_retention.png
                                                    #   notebooks/utility_retention.csv
python -m src.summary_figures --with_detail         # …and refresh the report-tier figures
```

- **`fig_headline.png`** — one operating point per family in tradeoff space, direct-labelled
  with its config. POINTS ONLY, and that is the point: both axes are metrics, so a line
  between two configs would trace the hidden σ, and where leakage is non-monotonic in σ it
  folds back and reads as two branches of a curve that does not exist.
- **`fig_sweeps.png`** — what each family does as σ rises, with σ ON the x-axis, so detection
  and leakage are each a genuine function of x and a line chart is honest. Columns are
  Input / Output / Output-LL; rows are detection and leakage. DP-SGD has no σ axis (its runs
  are a clip × noise grid) and lives in `dpsgd_grid.png`.
- **`utility_retention.png` / `utility_retention.csv`** — the supervisor's criterion
  ("preserve as much utility as possible for as long as it is safe"): for every privacy bar
  *t*, the best PR-AUC each family still reaches while leaking ≤ *t*. The curves use point
  estimates; the stricter "whole 95% interval clears the bar" standard is a column block in
  the CSV (`first_demonstrably_safe_bar` computes the number the write-up quotes for it).
  **Report appendix, not a slide** — it answers the question precisely and needs the CSV
  beside it, which is exactly what a slide cannot carry. Hand over the CSV when a bar is
  named.

The report tier behind them: `privacy_utility_mi_report.png` (all configs + CIs, faceted
one panel per family, the same Pareto staircase repeated in each), `dpsgd_grid.png`,
`privacy_utility_epsilon.png` (the ε *magnitudes* as a range chart — DP-SGD's ε < 0.2
against the perturbation families' 1e4–1e11, which is the formal-vs-empirical gap in one
glance), `pr_curves.png`, `pr_curves_by_type.png` (average precision as a grouped bar
chart, carrying the plateau blind spot), `timing_by_family.png`.

`fig_headline.png` and `privacy_utility_mi_report.png` share the same x-window and wording,
so a point picked off the headline sits at the same horizontal position in the report figure.
The operating points come from `operating_points.csv` — nothing is hardcoded in the plotting
code.

Two of them are deliberately *not* the row shown in the decision table below. For a family
that never reaches the privacy bar, that table falls back to its most private config —
which, for a family whose leakage barely moves with σ, is also close to its worst on
utility (Input σ=0.01 leaks 0.575 for PR-AUC 0.200, against σ=2.0's 0.585 for 0.542).
Anchoring the figures there would caricature the family, so the fallback row is kept only
when the leakage it bought is actually demonstrated — a paired Δeff-MI CI excluding zero.
That keeps Output-LL at σ=0.05, the study's one established privacy gain, and moves Input
to σ=2.0 and DP-SGD to C=0.5/nm=1 (their own best-utility Pareto points, carried in the
table's `alt_*` columns).

## Figure style

Every figure is drawn through `src/plotstyle.py`, which owns the rcParams, the family
palette and the title/subtitle/caption helpers. Six rules hold across the whole set:

1. **Lines only where y is a function of x** — σ-sweeps, retention steps, PR curves. In
   tradeoff (leakage vs utility) space: points, never connected.
2. **One marker shape (a circle); colour is the only family encoding.** Hollow-vs-solid,
   stars and per-family shapes are retired — a reader should not have to learn a glyph
   legend. The facts they carried moved into captions and config labels.
3. **No sentences on the graph.** A figure carries a short noun-phrase title, axis labels
   (the glosses like `(0.5 = attacker learns nothing)` stay — they are labels, not prose)
   and short point/series labels. Takeaway-sentence titles, subtitle sentences and bottom
   caption paragraphs all live in `Figure_Explainer.docx`. `plotstyle.style_figure`
   **raises** on a caption longer than `CAPTION_CLAUSE_MAX`, so the text cannot creep back
   one clause at a time the way it did last time. Two report figures keep a single short
   exclusion clause; nothing else keeps any.
4. **Reference lines carry one-word end tags** — `chance`, `baseline`, lowercase, from
   `plotstyle.CHANCE_LABEL` / `BASELINE_LABEL`. Tagging is the DEFAULT; a figure opts out
   (`label=None`) only for repeat panels on a shared axis, where the tag is printed once in
   the first panel and read across.
5. **Direct-label when series ends separate; use a small legend when they converge.**
   `utility_retention`'s curves meet at both right-hand clusters and `pr_curves`' overlap
   for most of their length, so those use `plotstyle.small_legend`; everything else labels
   in place.
6. **Error bars on the report-tier figure only.** Presentation-tier figures leave the
   uncertainty to `privacy_utility_mi_report.png`, which is faceted per family precisely so
   the intervals are legible rather than a hairball.

The five family hues are checked with a CVD/contrast validator; the closest pair sits in
the band that is legal only alongside a secondary encoding, so every figure drawing both
also direct-labels or facets them. Re-validate if you retune them.

### Personalized is excluded from every figure, and kept in every table

`Personalized` is a single-client *simulation* of per-user DP: no formal ε, and
mechanically it is input perturbation, which already has its own series. On a
privacy–utility axis it read as a sixth independent defence while being neither
independent nor, at this scale, a defence. So it is dropped from the figures
(`anchors.FIGURE_EXCLUDED`, applied at the one place each figure picks its families) and
**retained everywhere else** — `results_table.csv`, `operating_points.csv`,
`utility_retention.csv`, `src/personalized.py` — because it is the bridge to the federated
phase and gets a sentence in the write-up there.

<!-- BEGIN operating-points (generated by src.final_comparison) -->
## Decision view: best config per family under a privacy bar

| Family | Config | eff-MI | PR-AUC | ΔPR-AUC vs baseline (95% paired CI) | Δeff-MI vs baseline (95% paired CI) |
|---|---|---|---|---|---|
| Input ⚠︎ does not reach bar | σ=0.01 | 0.575 | 0.200 | **-0.383 [-0.440, -0.289]** | -0.111 [-0.181, +0.028] (n.s.) |
| Output | σ=0.0091809 | 0.507 | 0.556 | **-0.054 [-0.070, -0.028]** | -0.144 [-0.247, +0.073] (n.s.) |
| Output (last-layer) ⚠︎ does not reach bar | σ=0.05 | 0.627 | 0.593 | **-0.015 [-0.031, -0.001]** | **-0.073 [-0.110, -0.008]** |
| DP-SGD (fixed-clip baseline) ⚠︎ does not reach bar | C=0.5, nm=2 | 0.562 | 0.189 | **-0.399 [-0.445, -0.308]** | -0.118 [-0.204, +0.052] (n.s.) |
| Personalized ⚠︎ does not reach bar | σ=v1:0.104 \| v2:0.097 \| v3:0.056 | 0.671 | 0.215 | **-0.376 [-0.429, -0.284]** | -0.028 [-0.061, +0.029] (n.s.) |

- ⚠︎ **Input** — does not reach bar (eff-MI ≤ 0.55); within-family Pareto point is σ=2 (PR 0.542, eff-MI 0.585)
- ⚠︎ **Output (last-layer)** — does not reach bar (eff-MI ≤ 0.55); within-family Pareto point is σ=0.0052183 (PR 0.612, eff-MI 0.722)
- ⚠︎ **DP-SGD (fixed-clip baseline)** — does not reach bar (eff-MI ≤ 0.55); within-family Pareto point is C=0.5, nm=1 (PR 0.218, eff-MI 0.566)
- ⚠︎ **Personalized** — does not reach bar (eff-MI ≤ 0.55)

*Privacy bar: effective MI-AUC ≤ 0.55 (0.5 = chance). **Bold** = 95% paired CI excludes zero; `(n.s.)` = interval crosses zero, so the difference is not established. Negative Δeff-MI = less leakage. A flagged family's row is its config CLOSEST to the bar, which is not necessarily its best-utility choice — see the per-family notes above. CIs are paired session bootstraps — both models are scored on the SAME resampled sessions, not compared by checking whether separate CIs overlap.*
<!-- END operating-points -->
