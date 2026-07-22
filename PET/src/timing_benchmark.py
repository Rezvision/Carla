"""
Timing benchmark: clean, same-machine sec/epoch per family (TIMING_TASK.md parts C+D).

Why a benchmark at all when part A already estimates cost from history? Because the
historical numbers are upper bounds contaminated by evaluation, early stopping means each
model ran a different number of epochs, and the sweep spanned days. Here every family runs
the SAME fixed 3 epochs on the SAME data on the SAME machine, so sec/epoch is directly
comparable and the DP-SGD slowdown factor is a real ratio.

Design points that matter for the numbers being honest:
  - Fixed 3 epochs, early stopping disabled (patience > epochs). Cost per epoch is the
    quantity of interest; how many epochs a config happens to need is a separate axis
    (reported from history in the summary table).
  - The first config is run TWICE and only the second measurement is kept. CUDA context
    creation, cuDNN autotuning and the first pass over the data land on whichever config
    runs first and would otherwise be charged to it.
  - Checkpoints go to a scratch directory, never notebooks/models — a benchmark must not
    overwrite the trained artefacts the published results came from.
  - Output families are derived, not trained: their cost is a noise application on an
    existing checkpoint, measured directly and reported as ~0 rather than omitted.
"""
import argparse
import os
import shutil
import tempfile

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.compare import (  # noqa: E402
    CLIP_RANGE, DELTA, _fresh_model, prepare_carla_data, save_figure,
)
from src import plotstyle  # noqa: E402
from src.perturbation import apply_input_noise, apply_output_noise, apply_output_noise_last_layer  # noqa: E402
from src.timing import device_label, hostname, time_block  # noqa: E402
from src.train import train  # noqa: E402

# Fixed cost probe. 3 epochs is enough to average out per-epoch jitter while keeping the
# whole benchmark to order-of-minutes on a GPU.
BENCHMARK_EPOCHS = 3
BENCHMARK_INPUT_SIGMA = 0.1   # mid-sweep input σ — cost is σ-independent, value is arbitrary
BENCHMARK_DPSGD_CLIP = 1.0
BENCHMARK_DPSGD_NM = 1.0


def _train_once(train_ds, val_ds, scratch_dir, tag, epochs, batch_size, device):
    """One fixed-length train() run with early stopping disabled."""
    _, timing = train(
        _fresh_model(), train_ds, val_ds,
        save_path=os.path.join(scratch_dir, f"{tag}.pth"),
        epochs=epochs, batch_size=batch_size, device=device,
        patience=epochs + 1,  # never early-stop: we want exactly `epochs` epochs
    )
    return timing


def run_benchmark(data_dir, out_dir="notebooks", device=None, epochs=BENCHMARK_EPOCHS,
                  batch_size=256, dpsgd_physical_batch=512):
    """Measure sec/epoch per family and write timing_benchmark.csv + timing_by_family.png."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    scratch_dir = tempfile.mkdtemp(prefix="pet_timing_")  # never notebooks/models

    print("Loading sessions …")
    data = prepare_carla_data(data_dir)
    train_ds, val_ds = data["train_ds"], data["val_ds"]

    dev_label, host = device_label(device), hostname()
    print(f"Benchmark on {host} / {dev_label}, {epochs} epochs, batch_size={batch_size}")

    rows = []

    def record(family, config, timing=None, derive_seconds=None, note=""):
        rows.append({
            "family": family, "config": config,
            "epochs": epochs if timing is not None else np.nan,
            "train_seconds": timing.train_seconds if timing is not None else np.nan,
            "sec_per_epoch": timing.sec_per_epoch if timing is not None else 0.0,
            "derive_seconds": derive_seconds if derive_seconds is not None else np.nan,
            "trained": timing is not None,
            "device": dev_label, "hostname": host, "batch_size": batch_size,
            "note": note,
        })

    try:
        # --- warm-up: the first config absorbs CUDA init + first data pass. Run baseline
        # twice and keep only the second measurement (the first is discarded, not reported).
        print("\n=== Warm-up (discarded): baseline ===")
        _train_once(train_ds, val_ds, scratch_dir, "warmup", epochs, batch_size, device)

        print("\n=== Baseline (measured) ===")
        record("Baseline", "no DP", _train_once(train_ds, val_ds, scratch_dir, "baseline",
                                                epochs, batch_size, device),
               note="warm measurement; the discarded warm-up run absorbed CUDA init")

        print(f"\n=== Input perturbation (σ={BENCHMARK_INPUT_SIGMA}) ===")
        noisy_train = apply_input_noise(train_ds, sigma=BENCHMARK_INPUT_SIGMA,
                                        clip_range=CLIP_RANGE)
        record("Input", f"σ={BENCHMARK_INPUT_SIGMA}",
               _train_once(noisy_train, val_ds, scratch_dir, "input", epochs, batch_size, device),
               note="noise is applied to the data once, then training is ordinary — so "
                    "cost/epoch matches the baseline")

        print("\n=== Output perturbation (derived — no training) ===")
        # Needs a trained model to derive from; the benchmark baseline checkpoint serves.
        base = _fresh_model().to(device)
        base.load_state_dict(torch.load(os.path.join(scratch_dir, "baseline.pth"),
                                        map_location=device, weights_only=True))
        with time_block(sync_device=device) as t_full:
            apply_output_noise(base, sigma=0.005)
        record("Output", "σ=0.005 (derived)", derive_seconds=t_full.seconds,
               note="derived from an existing checkpoint by seeded noise — no training")
        with time_block(sync_device=device) as t_ll:
            apply_output_noise_last_layer(base, sigma=0.005)
        record("Output (last-layer)", "σ=0.005 (derived)", derive_seconds=t_ll.seconds,
               note="derived from an existing checkpoint by seeded noise — no training")

        print(f"\n=== DP-SGD (clip={BENCHMARK_DPSGD_CLIP}, nm={BENCHMARK_DPSGD_NM}) ===")
        try:
            from src.train_dpsgd import train_dpsgd
        except ImportError:
            print("  Opacus not installed — skipping DP-SGD row")
        else:
            _, _, _, dp_timing = train_dpsgd(
                _fresh_model(), train_ds, val_ds,
                save_path=os.path.join(scratch_dir, "dpsgd.pth"),
                noise_multiplier=BENCHMARK_DPSGD_NM, max_grad_norm=BENCHMARK_DPSGD_CLIP,
                delta=DELTA, epochs=epochs, batch_size=batch_size,
                max_physical_batch_size=dpsgd_physical_batch, device=device,
                patience=epochs + 1,
            )
            record("DP-SGD", f"clip={BENCHMARK_DPSGD_CLIP}, nm={BENCHMARK_DPSGD_NM}",
                   dp_timing,
                   note="Opacus per-sample gradients + physical-batch splitting — the "
                        "cause of the slowdown vs baseline")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "timing_benchmark.csv")
    df.to_csv(path, index=False)
    print(f"\nBenchmark written to {path}")
    print(df[["family", "config", "sec_per_epoch", "derive_seconds"]].to_string(index=False))

    _plot_timing_by_family(df, figures_dir, epochs=epochs, batch_size=batch_size)
    write_summary_markdown(df, out_dir)
    return df


def replot(out_dir="notebooks", epochs=BENCHMARK_EPOCHS, batch_size=256):
    """
    Redraw timing_by_family.png from the committed timing_benchmark.csv.

    The measurement is committed data — it needs the CARLA set and a GPU to produce and is
    not reproducible on another machine anyway, since it is a wall-clock number for one
    device. Restyling the figure must not require re-measuring it, so the plot is separable
    from the benchmark that feeds it.
    """
    path = os.path.join(out_dir, "timing_benchmark.csv")
    df = pd.read_csv(path)
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    _plot_timing_by_family(df, figures_dir, epochs=epochs, batch_size=batch_size)
    return df


def _slowdown_vs_baseline(df):
    """DP-SGD sec/epoch ÷ baseline sec/epoch — the headline ratio. NaN if either is absent."""
    trained = df[df["trained"]]
    base = trained[trained["family"] == "Baseline"]["sec_per_epoch"]
    dp = trained[trained["family"] == "DP-SGD"]["sec_per_epoch"]
    if base.empty or dp.empty or float(base.iloc[0]) == 0:
        return float("nan")
    return float(dp.iloc[0]) / float(base.iloc[0])


# The width the timing figure's in-axes text is sized at; the report copy scales to it.
TIMING_DESIGN_WIDTH_IN = 9.5
# Deliberately tall for its width. save_figure preserves aspect when it re-exports at 6.5in,
# and the caption wraps to more lines on a narrower canvas — so a short figure loses a much
# larger FRACTION of its height to text in the report copy than at design size, and five
# rows of two-line tick labels end up overprinting. The extra height is that headroom.
TIMING_DESIGN_HEIGHT_IN = 5.2


def _short_device(dev):
    """
    'cuda:NVIDIA GeForce RTX 3070 Ti' -> 'RTX 3070 Ti'.

    The CSV records the device verbatim, which is right for the record and wrong for a
    subtitle: the vendor prefixes are the same on every row, so they carry no information
    and cost a third of the line. The model number is the part that identifies the machine.
    """
    text = str(dev).split(":", 1)[-1].strip()
    for noise in ("NVIDIA ", "GeForce "):
        text = text.replace(noise, "")
    # A device recorded as a bare index ('cuda:0') has no name to shorten TO — dropping the
    # prefix would leave a subtitle reading '0'. Keep the original whenever what survives
    # is not a name.
    if not text or text.isdigit():
        return str(dev)
    return text


def _plot_timing_by_family(df, figures_dir, epochs, batch_size):
    """
    Horizontal bars of sec/epoch by family.

    Bars are the right geom here (unlike the privacy-utility figures): time has a true
    zero and the message IS the ratio between families. Derived families are drawn as a
    zero-length bar with an explicit "≈0" label so their absence reads as a finding
    rather than as missing data.
    """
    order = ["Baseline", "Input", "DP-SGD", "Output", "Output (last-layer)"]
    sub = df.set_index("family").reindex([f for f in order if f in set(df["family"])]).reset_index()

    # The benchmark's family names are the short ones; the palette is keyed by the
    # results-table method names, so map the one that differs.
    palette_key = {"DP-SGD": plotstyle.DPSGD_METHOD}
    colors = [plotstyle.family_color(palette_key.get(f, f)) for f in sub["family"]]

    fig, ax = plt.subplots(figsize=(TIMING_DESIGN_WIDTH_IN, TIMING_DESIGN_HEIGHT_IN))
    y = np.arange(len(sub))
    vals = sub["sec_per_epoch"].fillna(0.0).to_numpy(dtype=float)
    ax.barh(y, vals, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f}\n{c}" for f, c in zip(sub["family"], sub["config"])], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("seconds per epoch   (lower = cheaper)")
    # Bars run horizontally here, so the rule that helps is a VERTICAL one — the opposite
    # of the study default. This is the one figure that overrides it.
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")

    slowdown = _slowdown_vs_baseline(df)
    span = max(vals.max(), 1e-9)
    value_labels = []
    for yi, (_, row) in zip(y, sub.iterrows()):
        if not row["trained"]:
            # Short form: "derived from a checkpoint" is the doc's job. What has to stay on
            # the bar is that the zero is a real measurement, not missing data \u2014 which
            # "derived" plus the actual milliseconds carries in four words.
            value_labels.append(ax.text(
                span * 0.01, yi,
                f"  \u22480 (derived, {row['derive_seconds']*1000:.1f} ms)",
                va="center", fontsize=8, color=plotstyle.MUTED))
            continue
        label = f"  {row['sec_per_epoch']:.1f} s/epoch"
        if row["family"] == "DP-SGD" and np.isfinite(slowdown):
            label += f"   ({slowdown:.1f}\u00d7 baseline)"
        value_labels.append(
            ax.text(row["sec_per_epoch"] + span * 0.01, yi, label, va="center", fontsize=8.5))
    ax.set_xlim(0, span * 1.42)

    dev = _short_device(df["device"].iloc[0] if len(df) else "unknown")

    # The subtitle survives here, alone in the study, and only as a UNITS line: a wall-clock
    # number means nothing without the epochs it averages over and the device it ran on, so
    # these two facts are part of reading the axis rather than commentary on it. Everything
    # else \u2014 the host, why Opacus is 58x (per-sample gradients), why the output families
    # train nothing \u2014 is prose and lives in the doc.
    title = "Training cost by defence family"
    subtitle = f"{epochs} epochs  \u00b7  {dev}"

    # Everything INSIDE the axes has to shrink with the canvas. This figure is five rows of
    # two-line tick labels beside bars carrying their own value labels, all in points: at
    # report width the tick labels overprint each other and "(58.0x baseline)" runs off the
    # right edge. Sizes are captured here, before any scaling, so relayout never compounds.
    scaled = value_labels + list(ax.get_yticklabels())
    base_sizes = [t.get_fontsize() for t in scaled]

    def relayout(f):
        plotstyle.scale_text_sizes(scaled, base_sizes, f.get_size_inches()[0],
                                   TIMING_DESIGN_WIDTH_IN)
        plotstyle.style_figure(f, title, subtitle)

    relayout(fig)
    save_figure(fig, figures_dir, "timing_by_family.png", relayout=relayout)
    plt.close(fig)


# Typical epochs-to-early-stop observed in the committed runs. Taken from the training
# logs rather than re-derived here (the benchmark deliberately disables early stopping,
# so it cannot measure this itself).
_TYPICAL_EPOCHS = {
    "Baseline": "~20–30",
    "Input": "16–20 (σ=0.6–2.0 stopped at 18/16/20)",
    "DP-SGD": "~30 (rarely early-stops)",
    "Output": "n/a (derived)",
    "Output (last-layer)": "n/a (derived)",
}

_MARGINAL_COST = {
    "Baseline": "n/a (single model)",
    "Input": "full retrain per σ",
    "DP-SGD": "full retrain per (clip, nm) cell",
    "Output": "~0 — re-derived from the baseline checkpoint at eval time",
    "Output (last-layer)": "~0 — re-derived from the baseline checkpoint at eval time",
}


def write_summary_markdown(bench_df, out_dir="notebooks", retro_path=None):
    """
    Part D: one markdown row per family for the report.

    Joins the clean benchmark sec/epoch with the historical epochs-to-early-stop and the
    retrospective per-model range, so the reader can convert "cost per epoch" into "cost of
    adding one more σ" — the question that actually decides which defence is affordable.
    """
    retro_path = retro_path or os.path.join(out_dir, "timing_retrospective.csv")
    retro_range = {}
    if os.path.exists(retro_path):
        retro = pd.read_csv(retro_path)
        retro = retro[retro["source"] == "mtime-delta"]
        # CARLA only: this benchmark measures the CARLA GRU autoencoder. The Kaggle rows
        # are a much smaller tabular MLP with sub-second runs, so pooling the two would
        # report a range like "0–27 min" whose lower end belongs to a different model on a
        # different dataset.
        if "dataset" in retro.columns:
            retro = retro[retro["dataset"] == "CARLA"]
        for fam, grp in retro.groupby("family"):
            vals = pd.to_numeric(grp["est_minutes"], errors="coerce").dropna()
            if len(vals):
                retro_range[fam] = f"{vals.min():.0f}–{vals.max():.0f} min"

    slowdown = _slowdown_vs_baseline(bench_df)
    lines = [
        "# Training cost by DP family",
        "",
        f"Machine: `{bench_df['hostname'].iloc[0]}` · device: `{bench_df['device'].iloc[0]}` · "
        f"batch size: {int(bench_df['batch_size'].iloc[0])} · "
        f"benchmark: {int(bench_df['epochs'].dropna().iloc[0])} fixed epochs, early stopping disabled.",
        "",
        "| Family | sec/epoch (benchmark) | Typical epochs to early-stop | Marginal cost of one more σ | Retrospective per-model estimate |",
        "|---|---|---|---|---|",
    ]
    order = ["Baseline", "Input", "DP-SGD", "Output", "Output (last-layer)"]
    retro_key = {"Output": "Output", "Output (last-layer)": "Output (last-layer)"}
    for fam in order:
        row = bench_df[bench_df["family"] == fam]
        if row.empty:
            continue
        row = row.iloc[0]
        if row["trained"]:
            sec = f"{row['sec_per_epoch']:.1f}"
            if fam == "DP-SGD" and np.isfinite(slowdown):
                sec += f" ({slowdown:.1f}× baseline)"
        else:
            sec = f"≈0 — {row['derive_seconds']*1000:.1f} ms to derive (no training)"
        lines.append(
            f"| {fam} | {sec} | {_TYPICAL_EPOCHS.get(fam, '—')} | {_MARGINAL_COST.get(fam, '—')} "
            f"| {retro_range.get(retro_key.get(fam, fam), 'n/a (derived at eval time)' if not row['trained'] else '—')} |"
        )
    lines += [
        "",
        "Notes:",
        "- Benchmark numbers are wall clock over a fixed 3-epoch run with early stopping "
        "disabled, so families are directly comparable per epoch.",
        "- DP-SGD's slowdown comes from Opacus computing **per-sample** gradients and "
        "splitting the logical batch into smaller physical batches to fit memory.",
        "- Output / output-last-layer models are **derived** from the baseline checkpoint by "
        "seeded noise, so an extra σ costs an evaluation, not a training run. That asymmetry "
        "is why the output sweeps could be densified to 12 σ for free while the input sweep "
        "needed three fresh trainings.",
        "- Retrospective estimates are upper bounds from checkpoint mtime deltas (they "
        "include the evaluation that ran between trainings); see `timing_retrospective.csv`.",
    ]
    path = os.path.join(out_dir, "timing_summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Summary table written to {path}")
    return path


if __name__ == "__main__":
    import sys

    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/CARLA_processed")
    parser.add_argument("--out_dir", default="notebooks")
    parser.add_argument("--epochs", type=int, default=BENCHMARK_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replot", action="store_true",
                        help="Redraw the figure from the committed timing_benchmark.csv "
                             "without re-running the benchmark (no GPU or dataset needed).")
    args = parser.parse_args()
    if args.replot:
        replot(args.out_dir, epochs=args.epochs, batch_size=args.batch_size)
    else:
        run_benchmark(args.data_dir, args.out_dir, epochs=args.epochs,
                      batch_size=args.batch_size)
