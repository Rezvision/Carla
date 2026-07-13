"""
Tests for the Kaggle tabular dataset path.
All tests use the real CSV file at data/synthetic_telemetry_data.csv.
We use pytest.importorskip / pytest.mark.skipif to skip gracefully if the file
is absent (e.g. in CI without the data), but locally the file must be present.
"""
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

DATA_PATH = "data/synthetic_telemetry_data.csv"


@pytest.fixture(scope="module")
def raw_df():
    """Load the full Kaggle CSV once per test module."""
    from src.kaggle_data import load_kaggle
    try:
        return load_kaggle(DATA_PATH)
    except FileNotFoundError:
        pytest.skip(f"Kaggle data not found at {DATA_PATH}")


@pytest.fixture(scope="module")
def splits(raw_df):
    from src.kaggle_data import split_kaggle_stratified
    return split_kaggle_stratified(raw_df, seed=42)


@pytest.fixture(scope="module")
def datasets(splits):
    from src.kaggle_data import make_kaggle_dataset
    train_df, val_df, test_df = splits
    train_ds, train_labels, scaler = make_kaggle_dataset(train_df, fit_scaler=True)
    val_ds, val_labels, _ = make_kaggle_dataset(val_df, scaler=scaler)
    test_ds, test_labels, _ = make_kaggle_dataset(test_df, scaler=scaler)
    return {
        "train_ds": train_ds, "train_labels": train_labels,
        "val_ds": val_ds, "val_labels": val_labels,
        "test_ds": test_ds, "test_labels": test_labels,
        "scaler": scaler,
        "train_df": train_df,
    }


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

def test_kaggle_features_count():
    from src.kaggle_data import KAGGLE_FEATURES
    assert len(KAGGLE_FEATURES) == 32


def test_load_kaggle_columns(raw_df):
    from src.kaggle_data import KAGGLE_FEATURES
    for col in KAGGLE_FEATURES + ["anomaly_label", "vehicle_id"]:
        assert col in raw_df.columns, f"Missing column: {col}"


def test_load_kaggle_anomaly_binary(raw_df):
    vals = set(raw_df["anomaly_label"].unique())
    assert vals <= {0, 1}, f"anomaly_label contains values other than 0/1: {vals}"


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

def test_split_no_row_in_two_splits(splits):
    """No row (by original DataFrame index) appears in more than one split."""
    train_df, val_df, test_df = splits
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    assert train_idx.isdisjoint(val_idx), "train and val share rows"
    assert train_idx.isdisjoint(test_idx), "train and test share rows"
    assert val_idx.isdisjoint(test_idx), "val and test share rows"


def test_train_split_normal_only(splits):
    train_df, _, _ = splits
    assert train_df["anomaly_label"].sum() == 0, "Train split contains anomaly rows"


def test_all_failures_in_val_or_test(raw_df, splits):
    """Every failure row must be preserved in val or test (none discarded)."""
    total_failures = int(raw_df["anomaly_label"].sum())
    _, val_df, test_df = splits
    preserved = int(val_df["anomaly_label"].sum()) + int(test_df["anomaly_label"].sum())
    assert preserved == total_failures, (
        f"Expected {total_failures} failures across val+test, got {preserved}"
    )


def test_test_has_min_anomalies(splits):
    """Test split must have at least 5 anomalies for AUROC to be meaningful."""
    _, _, test_df = splits
    n_anomalies = int(test_df["anomaly_label"].sum())
    assert n_anomalies >= 5, (
        f"Test split has only {n_anomalies} anomalies — AUROC would be unreliable"
    )


# ---------------------------------------------------------------------------
# Dataset construction tests
# ---------------------------------------------------------------------------

def test_make_kaggle_dataset_shape(datasets):
    from src.kaggle_data import KAGGLE_FEATURES
    X = datasets["train_ds"].tensors[0]
    assert X.dim() == 2, f"Expected 2D tensor, got {X.dim()}D"
    assert X.shape[1] == len(KAGGLE_FEATURES), f"Expected {len(KAGGLE_FEATURES)} features"


def test_make_kaggle_dataset_scaler_reuse(splits):
    from src.kaggle_data import make_kaggle_dataset
    train_df, val_df, _ = splits
    _, _, scaler = make_kaggle_dataset(train_df, fit_scaler=True)
    _, _, returned = make_kaggle_dataset(val_df, scaler=scaler)
    assert returned is scaler, "Scaler returned for val should be the same object passed in"


def test_make_kaggle_labels_length(datasets):
    X = datasets["train_ds"].tensors[0]
    assert len(datasets["train_labels"]) == len(X), "Labels length must match tensor length"


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

def test_augment_increases_size(splits):
    from src.kaggle_data import make_kaggle_dataset
    train_df, _, _ = splits
    ds_orig, _, scaler = make_kaggle_dataset(train_df, fit_scaler=True, augment=False)
    ds_aug, _, _ = make_kaggle_dataset(train_df, scaler=scaler, augment=True, augment_sigma=0.01)
    assert len(ds_aug) > len(ds_orig), "Augmented dataset must be larger than original"


def test_augment_no_verbatim_duplicates(splits):
    from src.kaggle_data import make_kaggle_dataset
    train_df, _, _ = splits
    _, _, scaler = make_kaggle_dataset(train_df, fit_scaler=True, augment=False)
    ds_aug, _, _ = make_kaggle_dataset(train_df, scaler=scaler, augment=True, augment_sigma=0.01)
    X = ds_aug.tensors[0]
    n_orig = len(X) // 2
    orig = X[:n_orig]
    jitter = X[n_orig:]
    assert not torch.allclose(orig, jitter), "Jitter copies must differ from originals"


# ---------------------------------------------------------------------------
# MLPAutoencoder tests
# ---------------------------------------------------------------------------

def test_mlp_ae_output_shape():
    from src.model import MLPAutoencoder
    from src.kaggle_data import KAGGLE_FEATURES
    model = MLPAutoencoder(input_size=len(KAGGLE_FEATURES))
    x = torch.randn(8, len(KAGGLE_FEATURES))
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_mlp_ae_output_finite():
    from src.model import MLPAutoencoder
    from src.kaggle_data import KAGGLE_FEATURES
    model = MLPAutoencoder(input_size=len(KAGGLE_FEATURES))
    x = torch.randn(16, len(KAGGLE_FEATURES))
    out = model(x)
    assert torch.isfinite(out).all(), "Model output contains NaN or Inf"


def test_mlp_ae_has_output_layer():
    from src.model import MLPAutoencoder
    model = MLPAutoencoder(input_size=32)
    assert hasattr(model, "output_layer"), "MLPAutoencoder must have output_layer for perturbation compatibility"


# ---------------------------------------------------------------------------
# Evaluate compatibility with 2D tabular tensors
# ---------------------------------------------------------------------------

def test_reconstruction_errors_2d():
    from src.model import MLPAutoencoder
    from src.evaluate import reconstruction_errors
    model = MLPAutoencoder(input_size=9)
    ds = TensorDataset(torch.randn(50, 9))
    errors = reconstruction_errors(model, ds)
    assert errors.shape == (50,), f"Expected (50,), got {errors.shape}"
    assert (errors >= 0).all(), "Reconstruction errors must be non-negative"


def test_compute_metrics_labeled_keys():
    from src.model import MLPAutoencoder
    from src.evaluate import compute_metrics_labeled
    model = MLPAutoencoder(input_size=9)
    # Create a test set with some anomalies (need both classes for AUROC)
    normal = torch.randn(40, 9)
    anomaly = torch.randn(10, 9) * 5  # large values → high reconstruction error
    X = torch.cat([normal, anomaly])
    labels = np.array([0] * 40 + [1] * 10)
    ds = TensorDataset(X)
    metrics = compute_metrics_labeled(model, ds, labels)
    for key in ("auroc", "f1", "fpr_at_95tpr", "mean_mse_normal", "threshold"):
        assert key in metrics, f"Missing key: {key}"


def test_compute_metrics_labeled_auroc_range():
    from src.model import MLPAutoencoder
    from src.evaluate import compute_metrics_labeled
    model = MLPAutoencoder(input_size=9)
    normal = torch.randn(40, 9)
    anomaly = torch.randn(10, 9) * 5
    X = torch.cat([normal, anomaly])
    labels = np.array([0] * 40 + [1] * 10)
    ds = TensorDataset(X)
    metrics = compute_metrics_labeled(model, ds, labels)
    assert 0.0 <= metrics["auroc"] <= 1.0, f"AUROC out of range: {metrics['auroc']}"


def test_mi_auc_tabular():
    from src.model import MLPAutoencoder
    from src.evaluate import membership_inference
    model = MLPAutoencoder(input_size=9)
    member_ds = TensorDataset(torch.randn(30, 9))
    nonmember_ds = TensorDataset(torch.randn(30, 9))
    result = membership_inference(model, member_ds, nonmember_ds)
    assert 0.0 <= result["mi_auc"] <= 1.0
    assert "mi_accuracy" not in result
