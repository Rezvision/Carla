import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    FEATURES, WINDOW, load_sessions, split_sessions, make_dataset,
    load_sessions_with_vehicle_ids, split_sessions_for_mi, session_groups_for_windows,
)


def _make_fake_sessions(n_sessions=5, rows_per_session=200, seed=0):
    rng = np.random.default_rng(seed)
    sessions = []
    for _ in range(n_sessions):
        df = pd.DataFrame(rng.random((rows_per_session, len(FEATURES))), columns=FEATURES)
        sessions.append(df)
    return sessions


def test_load_sessions_returns_dataframes(tmp_path):
    sessions = _make_fake_sessions(3)
    for i, df in enumerate(sessions):
        df.to_parquet(tmp_path / f"session_{i}.parquet")
    loaded = load_sessions(str(tmp_path))
    assert len(loaded) == 3
    for df in loaded:
        assert list(df.columns) == FEATURES


def test_split_sessions_sizes():
    sessions = _make_fake_sessions(20)
    train, val, test = split_sessions(sessions, train_ratio=0.8, val_ratio=0.1)
    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2
    assert len(train) + len(val) + len(test) == 20


def test_split_sessions_no_overlap():
    sessions = _make_fake_sessions(10)
    train, val, test = split_sessions(sessions)
    train_ids = {id(s) for s in train}
    val_ids = {id(s) for s in val}
    test_ids = {id(s) for s in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_sessions_deterministic():
    sessions = _make_fake_sessions(10)
    t1, v1, s1 = split_sessions(sessions, seed=42)
    t2, v2, s2 = split_sessions(sessions, seed=42)
    assert [id(s) for s in t1] == [id(s) for s in t2]


def test_make_dataset_shape():
    sessions = _make_fake_sessions(3, rows_per_session=200)
    ds, scaler = make_dataset(sessions, fit_scaler=True)
    X = ds.tensors[0]
    expected_windows = sum(max(0, len(s) - WINDOW) for s in sessions)
    assert X.shape == (expected_windows, WINDOW, len(FEATURES))
    assert X.dtype == torch.float32


def test_scaler_fitted_on_train_only():
    sessions = _make_fake_sessions(10)  # needs ≥10 so val_size = int(10*0.1) = 1
    train, val, _ = split_sessions(sessions)
    _, scaler = make_dataset(train, fit_scaler=True)
    val_ds, _ = make_dataset(val, scaler=scaler)
    assert scaler.mean_ is not None


def test_make_dataset_val_uses_train_scaler():
    sessions = _make_fake_sessions(10)
    train, val, _ = split_sessions(sessions)
    _, scaler = make_dataset(train, fit_scaler=True)
    val_ds, returned_scaler = make_dataset(val, scaler=scaler)
    assert returned_scaler is scaler


# ---------------------------------------------------------------------------
# B1: lazy WindowDataset must be bit-identical to the original eager windowing
# ---------------------------------------------------------------------------

def _make_dataset_eager_reference(sessions, scaler, window):
    """
    Verbatim copy of the ORIGINAL eager make_dataset windowing logic (pre-B1), used
    only to verify the new lazy WindowDataset produces identical values.
    """
    windows = []
    for df in sessions:
        scaled = scaler.transform(df[FEATURES])
        for i in range(len(scaled) - window):
            windows.append(scaled[i : i + window])
    return torch.tensor(np.array(windows), dtype=torch.float32)


def test_make_dataset_lazy_matches_eager_reference():
    sessions = _make_fake_sessions(4, rows_per_session=120, seed=3)
    ds, scaler = make_dataset(sessions, fit_scaler=True)
    expected = _make_dataset_eager_reference(sessions, scaler, WINDOW)

    actual = ds.tensors[0]
    assert torch.equal(actual, expected)

    for idx in [0, 5, len(ds) - 1]:
        (item,) = ds[idx]
        assert torch.equal(item, expected[idx])


# ---------------------------------------------------------------------------
# Tests for Fix 1: auxiliary kwarg + no test-set leakage in vehicle partitions
# ---------------------------------------------------------------------------

def _make_indexed_sessions(n=10):
    """10 single-row DataFrames where df.iloc[0, 0] == original index."""
    return [pd.DataFrame([[float(i)] + [0.0] * (len(FEATURES) - 1)], columns=FEATURES)
            for i in range(n)]


def test_split_sessions_auxiliary_tracks_vehicle_ids():
    """Vehicle IDs must travel through the same permutation as sessions."""
    sessions = _make_indexed_sessions(10)
    vehicle_ids = ["v1"] * 4 + ["v2"] * 3 + ["v3"] * 3

    (train_s, val_s, test_s), (train_vids, val_vids, test_vids) = split_sessions(
        sessions, auxiliary=vehicle_ids
    )

    # For every (session, vid) pair in each split, the vid must match the original
    for df, vid in list(zip(train_s, train_vids)) + list(zip(val_s, val_vids)) + list(zip(test_s, test_vids)):
        original_idx = int(df.iloc[0, 0])
        assert vehicle_ids[original_idx] == vid, (
            f"Session originally at index {original_idx} (vehicle {vehicle_ids[original_idx]}) "
            f"was paired with wrong vehicle id '{vid}'"
        )


def test_no_test_session_in_vehicle_train_partitions():
    """Sessions in the global test set must not appear in any vehicle training partition."""
    sessions = _make_indexed_sessions(15)
    vehicle_ids = ["v1"] * 5 + ["v2"] * 5 + ["v3"] * 5

    (train_s, _, test_s), (train_vids, _, _) = split_sessions(sessions, auxiliary=vehicle_ids)

    # Build per-vehicle training dict exactly as compare.py does after Fix 1
    vehicle_train = {}
    for df, vid in zip(train_s, train_vids):
        vehicle_train.setdefault(vid, []).append(df)

    # No test session may appear in any vehicle's training partition
    for test_df in test_s:
        for vid, v_dfs in vehicle_train.items():
            for train_df in v_dfs:
                assert not test_df.equals(train_df), (
                    f"Test session found in vehicle '{vid}' training partition — "
                    "split leaks test data into training!"
                )


# ---------------------------------------------------------------------------
# split_sessions_for_mi / session_groups_for_windows (within-session MI holdout)
# ---------------------------------------------------------------------------

def _make_indexed_row_sessions(n_rows_list=(40,), seed=0):
    """
    Sessions where EVERY FEATURES column equals the row's own GLOBALLY unique index
    (continuing across sessions, never repeating). Since StandardScaler.transform is a
    single affine (hence injective) map applied over the whole concatenated dataset,
    two windows sharing a frame would share an exact scaled value — lets us test
    frame-disjointness without inverting the scale.
    """
    sessions = []
    next_idx = 0
    for n_rows in n_rows_list:
        idx = np.arange(next_idx, next_idx + n_rows, dtype=float)
        sessions.append(pd.DataFrame({col: idx for col in FEATURES}))
        next_idx += n_rows
    return sessions


def test_split_sessions_for_mi_no_shared_frames():
    window = 5
    sessions = _make_indexed_row_sessions(n_rows_list=(40, 60))
    train_parts, holdout_parts = split_sessions_for_mi(sessions, holdout_frac=0.25, window=window)

    train_ds, scaler = make_dataset(train_parts, fit_scaler=True, window=window)
    holdout_ds, _ = make_dataset(holdout_parts, scaler=scaler, window=window)

    train_vals = set(train_ds.tensors[0].flatten().tolist())
    holdout_vals = set(holdout_ds.tensors[0].flatten().tolist())
    assert train_vals.isdisjoint(holdout_vals), \
        "Train and holdout windows must never share a frame (original row)"


def test_split_sessions_for_mi_guard_gap_size():
    window = 5
    n_rows = 100
    sessions = _make_indexed_row_sessions(n_rows_list=(n_rows,))
    train_parts, holdout_parts = split_sessions_for_mi(sessions, holdout_frac=0.2, window=window)

    train_part, holdout_part = train_parts[0], holdout_parts[0]
    assert len(train_part) > 0 and len(holdout_part) > 0

    # Original row index is encoded directly in the (unscaled) FEATURES columns.
    last_train_idx = train_part.iloc[-1, 0]
    first_holdout_idx = holdout_part.iloc[0, 0]
    gap = first_holdout_idx - last_train_idx - 1  # rows discarded strictly between them
    assert gap >= window


def test_split_sessions_for_mi_holdout_from_same_session():
    """Holdout rows must be an exact suffix of the SAME training session, not drawn
    from any other session."""
    window = 5
    sessions = _make_fake_sessions(3, rows_per_session=100, seed=5)
    train_parts, holdout_parts = split_sessions_for_mi(sessions, holdout_frac=0.2, window=window)

    for original_df, holdout_df in zip(sessions, holdout_parts):
        tail = original_df.iloc[-len(holdout_df):] if len(holdout_df) > 0 else original_df.iloc[0:0]
        pd.testing.assert_frame_equal(
            holdout_df.reset_index(drop=True), tail.reset_index(drop=True)
        )


def test_session_groups_for_windows_matches_window_counts():
    window = 5
    sessions = _make_fake_sessions(4, rows_per_session=37, seed=1)
    groups = session_groups_for_windows(sessions, window=window)

    expected_counts = [max(0, len(s) - window) for s in sessions]
    assert len(groups) == sum(expected_counts)
    for i, count in enumerate(expected_counts):
        assert (groups == i).sum() == count
