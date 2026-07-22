"""
Retrospective training-cost ESTIMATES from artefacts that already exist
(TIMING_TASK.md part A). No compute, no retraining.

These are estimates, never measurements, and the output labels them as such. Two sources
are supported:

  mtime-delta : the gap between consecutive checkpoint mtimes in a sequential sweep. This
                is an UPPER BOUND on the second model's training time — the gap also
                contains the evaluation that ran between the two trainings (on CARLA that
                is a full metric suite with bootstrap CIs, which is not small), plus any
                data prep. It is also measured checkpoint-to-checkpoint, i.e. best-epoch
                to best-epoch, not start-to-end of training.
  log         : a wall-clock timestamp parsed out of a run log. Preferred where present,
                because it brackets the training itself.

IMPORTANT — only valid on the machine that trained the models. Git does not preserve
mtimes, so on any fresh clone notebooks/models/*.pth carry CLONE times and every
mtime-delta below would be meaningless. run() refuses to guess: it records the hostname
it ran on so a table produced on the wrong machine is identifiable after the fact.
"""
import argparse
import os
import re
from datetime import datetime, timezone

import pandas as pd

from src.timing import hostname

# Which family a checkpoint filename belongs to. Order matters: the dpsgd/personalized
# prefixes must be tested before the generic input/baseline ones.
_FAMILY_PATTERNS = [
    (re.compile(r"^dpsgd_clip(?P<clip>[\d.]+)_nm(?P<nm>[\d.]+)\.pth$"), "DP-SGD"),
    (re.compile(r"^personalized\.pth$"), "Personalized"),
    (re.compile(r"^input_sigma(?P<sigma>.+)\.pth$"), "Input"),
    (re.compile(r"^baseline\.pth$"), "Baseline"),
]

# ISO-ish timestamps a log might carry, e.g. "2026-07-08 19:58:03" or "2026-07-08T19:58:03".
_LOG_TS = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")

# An mtime delta is only an estimate of training time if the two checkpoints came from the
# SAME sequential sweep. A gap larger than this means the box sat idle between sessions
# (a later sweep run days afterwards), and the delta measures wall-clock calendar time, not
# compute — emitting it as an "upper bound" would be actively misleading. Set well above the
# slowest genuine run observed here (DP-SGD ≈ 11 h) so real runs are never discarded.
SESSION_GAP_MINUTES = 24 * 60


def classify(filename):
    """Map a checkpoint filename to (family, config-label). Unknown files → (None, None)."""
    for pattern, family in _FAMILY_PATTERNS:
        m = pattern.match(filename)
        if not m:
            continue
        gd = m.groupdict()
        if family == "DP-SGD":
            return family, f"clip={gd['clip']}, nm={gd['nm']}"
        if family == "Input":
            return family, f"σ={gd['sigma']}"
        return family, ""
    return None, None


def log_has_timestamps(log_path):
    """True if a log carries parseable wall-clock timestamps we could bracket runs with."""
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if _LOG_TS.search(line):
                    return True
                if i > 5000:  # these logs are epoch spam; a timestamp would appear early
                    break
    except OSError:
        return False
    return False


def checkpoint_estimates(models_dir, machine=None, log_paths=()):
    """
    Estimate per-model training minutes from consecutive checkpoint mtimes.

    Checkpoints are ordered by mtime (the order the sweep produced them). Each model's
    estimate is the gap from the PREVIOUS checkpoint, which brackets "everything that
    happened between the two saves" — hence an upper bound. The first checkpoint has no
    predecessor and therefore no estimate; it is still emitted, with the reason recorded,
    so the table is a complete inventory rather than a silently filtered one.
    """
    machine = machine or hostname()
    have_log_ts = any(log_has_timestamps(p) for p in log_paths)

    entries = []
    for name in os.listdir(models_dir) if os.path.isdir(models_dir) else []:
        if not name.endswith(".pth"):
            continue
        family, config = classify(name)
        if family is None:
            continue
        path = os.path.join(models_dir, name)
        entries.append({"model": name, "family": family, "config": config,
                        "mtime": os.path.getmtime(path)})
    entries.sort(key=lambda e: e["mtime"])

    rows = []
    for i, e in enumerate(entries):
        stamp = datetime.fromtimestamp(e["mtime"], tz=timezone.utc).isoformat()
        if i == 0:
            rows.append({
                "model": e["model"], "family": e["family"], "config": e["config"],
                "machine": machine, "est_minutes": float("nan"), "source": "none",
                "checkpoint_mtime_utc": stamp,
                "caveat": "first checkpoint in the sweep — no preceding mtime to difference "
                          "against, so no estimate is available",
            })
            continue
        gap_min = (e["mtime"] - entries[i - 1]["mtime"]) / 60.0
        if gap_min > SESSION_GAP_MINUTES:
            # Not a sequential run: the previous checkpoint belongs to an earlier session,
            # so this gap is idle calendar time. Emit the row (the inventory stays complete)
            # but with NO estimate rather than a number that would be read as compute.
            rows.append({
                "model": e["model"], "family": e["family"], "config": e["config"],
                "machine": machine, "est_minutes": float("nan"), "source": "none",
                "checkpoint_mtime_utc": stamp,
                "caveat": f"gap to previous checkpoint is {gap_min/60:.1f} h (> "
                          f"{SESSION_GAP_MINUTES/60:.0f} h) — a separate training session, "
                          "so the delta measures idle calendar time, not training",
            })
            continue
        rows.append({
            "model": e["model"], "family": e["family"], "config": e["config"],
            "machine": machine, "est_minutes": round(gap_min, 2), "source": "mtime-delta",
            "checkpoint_mtime_utc": stamp,
            "caveat": "UPPER BOUND — gap to the previous checkpoint also contains the "
                      "evaluation run between the two trainings (bootstrap CIs) and is "
                      "measured best-epoch to best-epoch, not train start to train end",
        })

    if not have_log_ts and rows:
        # Be explicit about why no row has source='log' rather than leaving it unexplained.
        for r in rows:
            r["caveat"] += "; run logs carry no wall-clock timestamps, so no log-based " \
                           "cross-check was possible"
    return pd.DataFrame(rows)


def run(out_dir="notebooks", models_dir=None, kaggle_models_dir=None):
    """Write notebooks/timing_retrospective.csv from whatever history exists on this box."""
    models_dir = models_dir or os.path.join(out_dir, "models")
    kaggle_models_dir = kaggle_models_dir or os.path.join(out_dir, "kaggle_models")
    logs = [os.path.join(out_dir, n) for n in
            ("run.log", "run.err", "kaggle_run.log", "kaggle_run.err")]

    carla = checkpoint_estimates(models_dir, log_paths=logs)
    if not carla.empty:
        carla.insert(3, "dataset", "CARLA")
    kaggle = checkpoint_estimates(kaggle_models_dir, log_paths=logs)
    if not kaggle.empty:
        kaggle.insert(3, "dataset", "Kaggle")

    df = pd.concat([d for d in (carla, kaggle) if not d.empty], ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "timing_retrospective.csv")
    df.to_csv(path, index=False)
    print(f"Retrospective estimates written to {path} ({len(df)} rows)")
    if not df.empty:
        print("\nNOTE: these are ESTIMATES, not measurements. mtime-delta rows are upper "
              "bounds (they include the eval between trainings).")
        print(df[["dataset", "family", "config", "est_minutes", "source"]].to_string(index=False))
    return df


if __name__ == "__main__":
    import sys

    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="notebooks")
    args = parser.parse_args()
    run(args.out_dir)
