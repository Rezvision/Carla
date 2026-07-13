import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# The 9 model inputs. NOTE: 'timestamp' is deliberately excluded — it is the
# ordering of the sequence, not a value the model tries to reconstruct.
FEATURES = ["speed_kmh", "battery_level", "throttle", "brake", "steering", "gear", "location_x", "location_y", "location_z"]

# Sequence length: the GRU sees WINDOW (=50) consecutive timesteps at a time.
WINDOW = 50

# Load each driving session's parquet as its own DataFrame. Sessions are kept
# separate (a list) so that windowing later never crosses a session boundary.
def load_sessions(data_dir):
    sessions = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(data_dir, fname), columns=FEATURES)
            sessions.append(df)
    return sessions

# Split at the SESSION level, not the row level: whole drives go to train/val/
# test. This stops windows from the same drive leaking across the splits.
def split_sessions(sessions, train_ratio=0.8, val_ratio=0.1, seed=42, auxiliary=None):
    """
    Split sessions into train/val/test at the session level.

    auxiliary: optional parallel list (e.g. vehicle_ids) that travels through the
    same permutation. When provided, returns ((train, val, test), (aux_train, aux_val,
    aux_test)) instead of the plain 3-tuple, so callers can track per-session metadata
    (e.g. vehicle ID) without re-splitting independently.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(sessions))
    train_size = int(len(sessions) * train_ratio)
    val_size = int(len(sessions) * val_ratio)
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]
    test_idx  = indices[train_size + val_size:]
    train = [sessions[i] for i in train_idx]
    val   = [sessions[i] for i in val_idx]
    test  = [sessions[i] for i in test_idx]
    # With an auxiliary list, apply the SAME shuffled indices to it — keeps every
    # session paired with its vehicle id through the split (no independent re-splitting,
    # which is what caused the earlier test-set leakage).
    if auxiliary is None:
        return train, val, test
    return (train, val, test), (
        [auxiliary[i] for i in train_idx],
        [auxiliary[i] for i in val_idx],
        [auxiliary[i] for i in test_idx],
    )

# Turn sessions into the model's actual training examples: overlapping
# fixed-length windows, each standardised (zero mean / unit variance per feature).
def load_sessions_with_vehicle_ids(data_dir):
    """
    Same sorted order as load_sessions; returns a parallel list of vehicle IDs.
    Filename pattern: 'vehicle_1_YYYYMMDD_HHMMSS.parquet' → vehicle_id "1".
    Use this instead of load_sessions when you need to track vehicle identity
    through the global split (avoids the re-split leakage bug).
    """
    sessions, vehicle_ids = [], []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".parquet"):
            sessions.append(pd.read_parquet(os.path.join(data_dir, fname), columns=FEATURES))
            vehicle_ids.append(fname.split("_")[1])
    return sessions, vehicle_ids


def load_sessions_by_vehicle(data_dir):
    """
    Return {vehicle_id: [df, ...]} where vehicle_id is parsed from filename.
    Filename pattern: 'vehicle_1_YYYYMMDD_HHMMSS.parquet' → vehicle_id "1".
    """
    by_vehicle = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".parquet"):
            vid = fname.split("_")[1]
            df = pd.read_parquet(os.path.join(data_dir, fname), columns=FEATURES)
            by_vehicle.setdefault(vid, []).append(df)
    return by_vehicle


class WindowDataset(torch.utils.data.Dataset):
    """
    Lazily slices fixed-length windows from a list of per-session scaled float32
    arrays, instead of eagerly materialising every overlapping window up front
    (peak memory ~1.6 GB vs ~14 GB eager on the full CARLA data, since each window
    duplicates up to WINDOW-1 rows shared with its neighbours).

    Bit-identical to the original eager implementation: each session is scaled in
    float64 (scaler.transform) and cast to float32 exactly once immediately after
    (before slicing — casting is elementwise, so doing it per-session gives the same
    values as casting one big concatenated float64 array at the very end); windows
    are produced in the same order (sessions in order, start index ascending, never
    crossing a session boundary); the count per session is the same
    (len(session) - window).
    """

    def __init__(self, arrays, window):
        self.arrays = arrays  # list of (len_i, F) float32 arrays, one per session
        self.window = window
        self._counts = [max(0, len(a) - window) for a in arrays]
        self._offsets = np.cumsum([0] + self._counts)

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        session_i = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        start = idx - self._offsets[session_i]
        window = self.arrays[session_i][start : start + self.window]
        return (torch.from_numpy(window),)

    @property
    def tensors(self):
        """
        Materialise the full (N, W, F) float32 tensor, built directly in float32
        (no float64 intermediate), for back-compat with `dataset.tensors[0]` callers.
        """
        n = len(self)
        f = self.arrays[0].shape[1] if self.arrays else 0
        out = np.empty((n, self.window, f), dtype=np.float32)
        pos = 0
        for arr, count in zip(self.arrays, self._counts):
            for i in range(count):
                out[pos] = arr[i : i + self.window]
                pos += 1
        return (torch.from_numpy(out),)


def split_sessions_for_mi(sessions, holdout_frac=0.1, window=WINDOW):
    """
    For each TRAINING session, carve off a same-session held-out tail to use as MI
    non-members, with a WINDOW-row guard gap discarded in between.

    Why within-session: if MI non-members come from a DIFFERENT session (or vehicle),
    the attack partly detects "which session's distribution is this" rather than "was
    this trained on" -- a confound with memorisation (cf. the evaluation-pitfalls point
    of Carlini et al. 2022). Splitting within each session removes that confound.

    Why the guard gap: with stride-1 windows, a held-out window immediately adjacent to
    a trained window overlaps it in nearly every frame even with zero literally-shared
    windows -- a near-duplicate leak. Discarding `window` rows between train_part and
    holdout_part guarantees no member window and non-member window share any frame
    (make_dataset never windows across a DataFrame boundary, so windowing train_parts
    and holdout_parts as separate session lists enforces this).

    Args:
        sessions: list of session DataFrames (pass the TRAINING split only).
        holdout_frac: fraction of each session's tail reserved for the holdout part.
        window: must match the window size used downstream by make_dataset.

    Returns:
        train_parts, holdout_parts: parallel lists of DataFrames, same order/length
        as `sessions`.
    """
    train_parts, holdout_parts = [], []
    for df in sessions:
        cut = int(len(df) * (1 - holdout_frac))
        train_cut = cut - window
        if train_cut >= window and (len(df) - cut) >= window:
            train_parts.append(df.iloc[:train_cut])
            holdout_parts.append(df.iloc[cut:])
        else:
            train_parts.append(df)      # too short to split; keep whole, no holdout
    return train_parts, holdout_parts


def session_groups_for_windows(sessions, window=WINDOW):
    """
    Return a 1-D array of session-index labels, one per window that make_dataset(
    sessions, window=window) would produce, in the same order. Mirrors WindowDataset's
    window-count-per-session logic (max(0, len(session)-window) windows per session)
    without needing to touch WindowDataset itself.

    Used as the resampling unit ("groups") for the cluster bootstrap CI
    (bootstrap_auc_ci in evaluate.py) — window order/count here must exactly match
    make_dataset's.
    """
    if not sessions:
        return np.array([], dtype=int)
    return np.concatenate([
        np.full(max(0, len(df) - window), i, dtype=int)
        for i, df in enumerate(sessions)
    ])


def make_dataset(sessions, scaler=None, fit_scaler=False, window=WINDOW):
    all_rows = pd.concat(sessions, ignore_index=True)
    if scaler is None:
        scaler = StandardScaler()
    # Fit the scaler on TRAINING data only, then reuse it for val/test — fitting on
    # all data would leak test-set statistics into training.
    if fit_scaler:
        scaler.fit(all_rows[FEATURES])

    # Scale each session (float64, via scaler.transform) and cast to float32 exactly
    # once, immediately — windows are then sliced lazily by WindowDataset rather than
    # eagerly materialised (see WindowDataset docstring for the bit-identity argument).
    arrays = [scaler.transform(df[FEATURES]).astype(np.float32) for df in sessions]

    return WindowDataset(arrays, window=window), scaler