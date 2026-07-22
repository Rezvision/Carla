"""
Shared wall-clock instrumentation (TIMING_TASK.md part B).

Cost is the third axis alongside privacy and utility: a defence that needs 6 GPU-hours
per σ is a different proposition from one that is a closed-form noise application on an
existing checkpoint, even at identical (ε, PR-AUC). These helpers keep the measurement
identical across train.py / train_dpsgd.py / the sweeps so the numbers are comparable.

Two kinds of cost are recorded, and they must not be conflated:
  - train_seconds  : a model that had to be TRAINED (baseline, input, DP-SGD, personalized).
  - derive_seconds : a model DERIVED from an existing checkpoint by applying seeded noise
                     (output, output-last-layer). Expected ~0 — that is the finding, so it
                     is recorded explicitly rather than left blank.
"""
import platform
import socket
import time
from typing import NamedTuple

import torch


class TrainTiming(NamedTuple):
    """Wall-clock cost of one training run.

    sec_per_epoch is derived (not measured separately) so it can never disagree with
    train_seconds / epochs_ran — the tests assert exactly that identity.
    """
    train_seconds: float
    epochs_ran: int

    @property
    def sec_per_epoch(self) -> float:
        return self.train_seconds / self.epochs_ran if self.epochs_ran else float("nan")

    def as_row(self) -> dict:
        """Timing columns for a results-table row."""
        return {
            "train_seconds": self.train_seconds,
            "epochs_ran": self.epochs_ran,
            "sec_per_epoch": self.sec_per_epoch,
        }


def device_label(device=None) -> str:
    """
    Human-readable device id for the results table, e.g. 'cuda:NVIDIA GeForce RTX 3070 Ti'.

    The GPU model matters for interpreting sec/epoch, so the name is carried in the table
    rather than assumed from the hostname.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            return f"cuda:{torch.cuda.get_device_name(device)}"
        except Exception:  # pragma: no cover - defensive; never fail a run over a label
            return "cuda"
    return "cpu"


def hostname() -> str:
    """Machine that produced the measurement (timings are only comparable within one)."""
    try:
        return socket.gethostname()
    except Exception:  # pragma: no cover
        return platform.node() or "unknown"


def run_meta(device=None) -> dict:
    """Machine/device columns attached to every timed row."""
    return {"device": device_label(device), "hostname": hostname()}


class time_block:
    """
    Context manager measuring wall-clock seconds of the enclosed block.

    Used for the derived-model path and the once-per-run eval timing, where there is no
    epoch loop to instrument. CUDA work is asynchronous, so callers that time GPU work
    should pass sync_device to make the measurement honest.

        with time_block() as t:
            noisy = apply_output_noise(model, sigma=s)
        derive_seconds = t.seconds
    """

    def __init__(self, sync_device=None):
        self._sync_device = sync_device
        self.seconds = float("nan")

    def _sync(self):
        dev = self._sync_device
        if dev is not None and torch.device(dev).type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(dev)

    def __enter__(self):
        self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.seconds = time.perf_counter() - self._start
        return False


# Timing columns, in table order. Kept here so compare/compare_kaggle/reeval all agree.
TIMING_COLUMNS = ["train_seconds", "epochs_ran", "sec_per_epoch", "derive_seconds",
                  "device", "hostname"]
TIMING_HEADERS = ["train_seconds", "epochs_ran", "sec_per_epoch", "derive_seconds",
                  "device", "hostname"]


def save_run_timing_meta(out_dir, eval_seconds_per_model, device=None, name="timing_run_meta.csv"):
    """
    Write the once-per-run eval cost next to the results table.

    Kept OUT of the per-model table on purpose: it is one measurement describing the
    evaluation harness (dominated by the bootstrap CIs), not a property of any single
    model, so repeating it on ~40 rows would imply 40 independent measurements.
    """
    import os

    import pandas as pd

    row = {
        "eval_seconds_per_model": eval_seconds_per_model,
        **run_meta(device),
        "note": "one full evaluate_model() call incl. bootstrap CIs; measured once per run",
    }
    path = os.path.join(out_dir, name)
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"Run timing meta saved to {path}")
    return path
