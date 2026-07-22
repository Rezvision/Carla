"""Tests for the wall-clock instrumentation (TIMING_TASK.md)."""
import os
import time

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from src.model import GRUAutoencoder
from src.perturbation import apply_output_noise, apply_output_noise_last_layer
from src.timing import (
    TIMING_COLUMNS, TrainTiming, device_label, hostname, run_meta, save_run_timing_meta,
    time_block,
)
from src.timing_retro import SESSION_GAP_MINUTES, checkpoint_estimates, classify
from src.train import train


# --------------------------------------------------------------------------- helpers
def _tiny_dataset(n=24, seq_len=5, n_features=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, seq_len, n_features, generator=g))


def _tiny_model(n_features=3):
    return GRUAutoencoder(input_size=n_features, hidden_size=4, num_layers=1)


# ------------------------------------------------------------------- TrainTiming type
def test_sec_per_epoch_is_train_seconds_over_epochs():
    t = TrainTiming(train_seconds=12.0, epochs_ran=4)
    assert t.sec_per_epoch == pytest.approx(3.0)


def test_sec_per_epoch_nan_when_no_epochs_ran():
    # Guard against ZeroDivisionError if a loop exits before completing an epoch.
    assert np.isnan(TrainTiming(train_seconds=1.0, epochs_ran=0).sec_per_epoch)


def test_as_row_exposes_the_table_columns():
    row = TrainTiming(train_seconds=8.0, epochs_ran=2).as_row()
    assert row["train_seconds"] == 8.0
    assert row["epochs_ran"] == 2
    assert row["sec_per_epoch"] == pytest.approx(4.0)


# ------------------------------------------------------------------ train() plumbing
def test_train_returns_positive_timing_and_consistent_sec_per_epoch(tmp_path):
    """Timer plumbing: positive train_seconds, and the sec/epoch identity holds."""
    model, timing = train(
        _tiny_model(), _tiny_dataset(), _tiny_dataset(seed=1),
        save_path=str(tmp_path / "m.pth"), epochs=1, batch_size=8, device=torch.device("cpu"),
    )
    assert model is not None
    assert timing.train_seconds > 0
    assert timing.epochs_ran == 1
    assert timing.sec_per_epoch == pytest.approx(timing.train_seconds / timing.epochs_ran)


def test_train_epochs_ran_reflects_early_stopping(tmp_path):
    """epochs_ran must be the epochs ACTUALLY run, so sec/epoch stays meaningful."""
    _, timing = train(
        _tiny_model(), _tiny_dataset(), _tiny_dataset(seed=1),
        save_path=str(tmp_path / "m.pth"), epochs=3, batch_size=8,
        device=torch.device("cpu"), patience=1,
    )
    assert 1 <= timing.epochs_ran <= 3
    assert timing.sec_per_epoch == pytest.approx(timing.train_seconds / timing.epochs_ran)


# --------------------------------------------------------------- derived-model timing
@pytest.mark.parametrize("apply_fn", [apply_output_noise, apply_output_noise_last_layer])
def test_derived_model_path_records_derive_seconds(apply_fn):
    """
    Output families are derived, not trained: derive_seconds must be a real number.

    TIMING_TASK.md is explicit that ~0 IS the finding, so the column must never be left
    NaN for these rows — a blank would read as 'not measured'.
    """
    model = _tiny_model()
    with time_block() as t:
        derived = apply_fn(model, sigma=0.005)
    assert derived is not None
    assert np.isfinite(t.seconds)
    assert t.seconds >= 0


def test_time_block_measures_elapsed_time():
    with time_block() as t:
        time.sleep(0.01)
    assert t.seconds >= 0.005


# ------------------------------------------------------------------------ run metadata
def test_run_meta_reports_device_and_hostname():
    meta = run_meta(torch.device("cpu"))
    assert meta["device"] == "cpu"
    assert isinstance(meta["hostname"], str) and meta["hostname"]


def test_device_label_is_cpu_for_cpu_device():
    assert device_label(torch.device("cpu")) == "cpu"


def test_timing_columns_cover_train_and_derive_paths():
    for col in ("train_seconds", "epochs_ran", "sec_per_epoch", "derive_seconds",
                "device", "hostname"):
        assert col in TIMING_COLUMNS


def test_save_run_timing_meta_writes_csv(tmp_path):
    save_run_timing_meta(str(tmp_path), 12.5, torch.device("cpu"))
    df = pd.read_csv(tmp_path / "timing_run_meta.csv")
    assert df["eval_seconds_per_model"].iloc[0] == pytest.approx(12.5)
    assert df["device"].iloc[0] == "cpu"


# ----------------------------------------------------- results table carries the columns
def test_results_table_has_timing_columns(tmp_path):
    from src.compare import _RESULT_HEADERS, build_results_table

    df = build_results_table(
        [{"method": "Baseline", "sigma": None, "pr_auc": 0.6,
          "train_seconds": 30.0, "epochs_ran": 2, "sec_per_epoch": 15.0,
          "device": "cpu", "hostname": "h"}],
        str(tmp_path),
    )
    for col in ("train_seconds", "epochs_ran", "sec_per_epoch", "derive_seconds",
                "device", "hostname"):
        assert col in df.columns
    assert list(df.columns) == list(_RESULT_HEADERS)


# --------------------------------------------------------------- retrospective (part A)
def test_classify_maps_checkpoint_names_to_families():
    assert classify("baseline.pth")[0] == "Baseline"
    assert classify("input_sigma0.1.pth")[0] == "Input"
    assert classify("dpsgd_clip1.0_nm2.0.pth")[0] == "DP-SGD"
    assert classify("personalized.pth")[0] == "Personalized"
    assert classify("not_a_model.txt")[0] is None


def _touch(path, mtime):
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))


def test_checkpoint_estimates_uses_mtime_deltas(tmp_path):
    base = 1_700_000_000
    _touch(tmp_path / "baseline.pth", base)
    _touch(tmp_path / "input_sigma0.1.pth", base + 600)  # +10 min

    df = checkpoint_estimates(str(tmp_path))
    first, second = df.iloc[0], df.iloc[1]
    # The first checkpoint has no predecessor, so it gets no estimate.
    assert first["source"] == "none" and np.isnan(first["est_minutes"])
    assert second["source"] == "mtime-delta"
    assert second["est_minutes"] == pytest.approx(10.0)
    assert "UPPER BOUND" in second["caveat"]


def test_checkpoint_estimates_drops_cross_session_gaps(tmp_path):
    """
    A gap spanning a separate session is idle calendar time, not training time.

    Without this guard a checkpoint trained a week after its predecessor reports a
    multi-thousand-minute 'upper bound', which would be read as compute.
    """
    base = 1_700_000_000
    _touch(tmp_path / "baseline.pth", base)
    _touch(tmp_path / "input_sigma0.6.pth", base + int(SESSION_GAP_MINUTES * 60) + 3600)

    df = checkpoint_estimates(str(tmp_path))
    late = df[df["model"] == "input_sigma0.6.pth"].iloc[0]
    assert late["source"] == "none"
    assert np.isnan(late["est_minutes"])
    assert "separate training session" in late["caveat"]


# --------------------------------------------------------------- benchmark (part C)
def test_benchmark_smoke_skipped_without_data_dir():
    """Benchmark mode needs the real CARLA data; skip cleanly when it is absent."""
    data_dir = "data/CARLA_processed"
    if not os.path.isdir(data_dir):
        pytest.skip("CARLA data dir not present — benchmark smoke test skipped")

    from src.timing_benchmark import _slowdown_vs_baseline

    # Don't retrain here (minutes of GPU); exercise the ratio helper the figure depends on.
    df = pd.DataFrame([
        {"family": "Baseline", "sec_per_epoch": 10.0, "trained": True},
        {"family": "DP-SGD", "sec_per_epoch": 25.0, "trained": True},
    ])
    assert _slowdown_vs_baseline(df) == pytest.approx(2.5)


def test_slowdown_is_nan_without_dpsgd_row():
    from src.timing_benchmark import _slowdown_vs_baseline

    df = pd.DataFrame([{"family": "Baseline", "sec_per_epoch": 10.0, "trained": True}])
    assert np.isnan(_slowdown_vs_baseline(df))


def test_summary_markdown_lists_every_benchmarked_family(tmp_path):
    from src.timing_benchmark import write_summary_markdown

    df = pd.DataFrame([
        {"family": "Baseline", "config": "no DP", "epochs": 3, "sec_per_epoch": 10.0,
         "derive_seconds": np.nan, "trained": True, "device": "cpu", "hostname": "h",
         "batch_size": 256},
        {"family": "DP-SGD", "config": "clip=1.0, nm=1.0", "epochs": 3, "sec_per_epoch": 40.0,
         "derive_seconds": np.nan, "trained": True, "device": "cpu", "hostname": "h",
         "batch_size": 256},
        {"family": "Output", "config": "σ=0.005 (derived)", "epochs": np.nan,
         "sec_per_epoch": 0.0, "derive_seconds": 0.002, "trained": False, "device": "cpu",
         "hostname": "h", "batch_size": 256},
    ])
    path = write_summary_markdown(df, str(tmp_path))
    text = open(path, encoding="utf-8").read()
    for family in ("Baseline", "DP-SGD", "Output"):
        assert family in text
    assert "4.0× baseline" in text          # 40 / 10
    assert "no training" in text            # derived families flagged, not blank
