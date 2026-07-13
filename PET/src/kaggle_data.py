"""
Loader, split, and dataset builder for the Kaggle synthetic telemetry dataset.

The dataset is NOT a usable time series (timestamps are shuffled, 0% of vehicles
are monotonically increasing), so we treat each row as an independent tabular sample.
No windowing; output tensor shape is (N, n_features).

Split strategy: vehicle-level — whole vehicles go to one split, never two.
Train set: NORMAL rows only (failure_type == "No Failure").
Val/test sets: ALL rows from their vehicles (both normal and failure rows).
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


# 32 numeric sensor features: all columns except the 9 excluded below.
# Excluded: vehicle_id, brand, timestamp, failure_date, failure_type,
#           engine_failure_imminent, brake_issue_imminent, battery_issue_imminent,
#           abs_fault_indicator
KAGGLE_FEATURES = [
    "odometer_reading",
    "engine_temp_c",
    "engine_rpm",
    "oil_pressure_psi",
    "coolant_temp_c",
    "fuel_level_percent",
    "fuel_consumption_lph",
    "engine_load_percent",
    "throttle_pos_percent",
    "air_flow_rate_gps",
    "exhaust_gas_temp_c",
    "vibration_level",
    "engine_hours",
    "brake_fluid_level_psi",
    "brake_pad_wear_mm",
    "brake_temp_c",
    "brake_pedal_pos_percent",
    "wheel_speed_fl_kph",
    "wheel_speed_fr_kph",
    "wheel_speed_rl_kph",
    "wheel_speed_rr_kph",
    "battery_voltage_v",
    "battery_current_a",
    "battery_temp_c",
    "alternator_output_v",
    "battery_charge_percent",
    "battery_health_percent",
    "vehicle_speed_kph",
    "ambient_temp_c",
    "humidity_percent",
    "gps_latitude",
    "gps_longitude",
]


def load_kaggle(path):
    """
    Read the synthetic telemetry CSV and return a DataFrame with:
      - KAGGLE_FEATURES columns (32 numeric sensor readings)
      - vehicle_id column (kept for vehicle-level splitting)
      - anomaly_label column (1 = any failure, 0 = "No Failure")

    The returned DataFrame has one row per original CSV row.
    """
    df = pd.read_csv(path)
    df["anomaly_label"] = (df["failure_type"] != "No Failure").astype(int)
    return df[["vehicle_id"] + KAGGLE_FEATURES + ["anomaly_label"]].copy()


def split_kaggle_by_vehicle(df, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split the dataset at the vehicle_id level so no vehicle appears in two splits.

    Returns:
        train_df : rows from train vehicles, NORMAL only (anomaly_label == 0)
        val_df   : ALL rows from val vehicles (normal + failures)
        test_df  : ALL rows from test vehicles (normal + failures)
    """
    vehicle_ids = sorted(df["vehicle_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(vehicle_ids)

    n = len(shuffled)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_vids = set(shuffled[:train_size])
    val_vids = set(shuffled[train_size:train_size + val_size])
    test_vids = set(shuffled[train_size + val_size:])

    train_df = df[df["vehicle_id"].isin(train_vids) & (df["anomaly_label"] == 0)].copy()
    val_df = df[df["vehicle_id"].isin(val_vids)].copy()
    test_df = df[df["vehicle_id"].isin(test_vids)].copy()

    return train_df, val_df, test_df


def split_kaggle_stratified(df, normal_ratios=(0.7, 0.15, 0.15), failure_val_ratio=0.5, seed=42):
    """
    Row-level class-stratified split. Defensible because timestamps are shuffled (not sequential).

    Normal rows (anomaly_label==0): split into train / val / test by normal_ratios.
    Failure rows (anomaly_label==1): split into val / test ONLY by failure_val_ratio.
      → No failures in train; model still trains on normal rows only.
      → All 35 failures are preserved for evaluation (~17–18 in test).

    The original DataFrame index is preserved so downstream tests can check row-level
    disjointness with .index comparisons.

    Trade-off vs vehicle-level split: rows from the same vehicle may appear in both
    train and test. Given shuffled timestamps there is no temporal leakage.
    """
    rng = np.random.default_rng(seed)
    normal = df[df["anomaly_label"] == 0]
    failures = df[df["anomaly_label"] == 1]

    # Permute positional indices within each class independently.
    n_norm = len(normal)
    n_fail = len(failures)
    norm_perm = rng.permutation(n_norm)
    fail_perm = rng.permutation(n_fail)

    na = int(n_norm * normal_ratios[0])
    nb = na + int(n_norm * normal_ratios[1])
    fv_end = int(n_fail * failure_val_ratio)

    train_df = normal.iloc[norm_perm[:na]]
    val_df = pd.concat([
        normal.iloc[norm_perm[na:nb]],
        failures.iloc[fail_perm[:fv_end]],
    ])
    test_df = pd.concat([
        normal.iloc[norm_perm[nb:]],
        failures.iloc[fail_perm[fv_end:]],
    ])
    return train_df.copy(), val_df.copy(), test_df.copy()


def make_kaggle_dataset(rows_df, scaler=None, fit_scaler=False,
                        augment=False, augment_sigma=0.01):
    """
    Build a tabular TensorDataset from a slice of the Kaggle DataFrame.

    Args:
        rows_df       : DataFrame with KAGGLE_FEATURES + anomaly_label columns
        scaler        : fitted StandardScaler (pass for val/test; None if fit_scaler=True)
        fit_scaler    : if True, fit a new StandardScaler on rows_df (train only)
        augment       : if True, append Gaussian-jittered copies of the rows (train only)
        augment_sigma : std of jitter noise

    Returns:
        dataset  : TensorDataset of shape (N, 32)  — 2D, no windowing
        labels   : np.ndarray of int (0=normal, 1=anomaly), length N
        scaler   : the StandardScaler used (fitted or passed-through)

    Augmentation is intentionally leakage-safe: it only runs on the training slice,
    which is already split off from val/test before this function is called.
    """
    labels = rows_df["anomaly_label"].values.astype(int)
    X = rows_df[KAGGLE_FEATURES].values.astype(np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)

    if augment:
        rng = np.random.default_rng(42)
        jitter = (X + rng.normal(0, augment_sigma, X.shape)).astype(np.float32)
        X = np.vstack([X, jitter])
        labels = np.concatenate([labels, np.zeros(len(jitter), dtype=int)])

    tensor = torch.tensor(X, dtype=torch.float32)
    return TensorDataset(tensor), labels, scaler
