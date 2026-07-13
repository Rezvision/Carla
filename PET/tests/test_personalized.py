import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from src.data import load_sessions_by_vehicle, FEATURES
from src.personalized import (
    personalized_noise_map,
    apply_personalized_input_noise,
    per_vehicle_mi,
)
from src.model import GRUAutoencoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vehicle_parquets(tmp_path, vehicles=("1", "2", "3"), sessions_per=3, rows=100):
    """Write fake parquet files named vehicle_<id>_YYYYMMDD_HHMMSS.parquet."""
    rng = np.random.default_rng(0)
    for vid in vehicles:
        for i in range(sessions_per):
            df = pd.DataFrame(
                rng.random((rows, len(FEATURES))), columns=FEATURES
            )
            fname = f"vehicle_{vid}_2026010{i}_120000.parquet"
            df.to_parquet(tmp_path / fname)


def _tiny_vehicle_datasets(n_per_vehicle=40, w=10, f=9):
    torch.manual_seed(0)
    return {
        "1": TensorDataset(torch.randn(n_per_vehicle, w, f)),
        "2": TensorDataset(torch.randn(n_per_vehicle, w, f) + 0.5),
        "3": TensorDataset(torch.randn(n_per_vehicle, w, f) - 0.5),
    }


# ---------------------------------------------------------------------------
# load_sessions_by_vehicle
# ---------------------------------------------------------------------------

def test_load_sessions_by_vehicle_keys(tmp_path):
    _make_vehicle_parquets(tmp_path)
    by_vehicle = load_sessions_by_vehicle(str(tmp_path))
    assert set(by_vehicle.keys()) == {"1", "2", "3"}


def test_load_sessions_by_vehicle_counts(tmp_path):
    _make_vehicle_parquets(tmp_path, sessions_per=3)
    by_vehicle = load_sessions_by_vehicle(str(tmp_path))
    for vid, sessions in by_vehicle.items():
        assert len(sessions) == 3, f"Vehicle {vid} should have 3 sessions"


def test_load_sessions_by_vehicle_columns(tmp_path):
    _make_vehicle_parquets(tmp_path)
    by_vehicle = load_sessions_by_vehicle(str(tmp_path))
    for vid, sessions in by_vehicle.items():
        for df in sessions:
            assert list(df.columns) == FEATURES


# ---------------------------------------------------------------------------
# personalized_noise_map
# ---------------------------------------------------------------------------

def test_personalized_noise_map_ordering():
    """Higher MI AUC → higher σ."""
    mi_aucs = {"1": 0.6, "2": 0.8, "3": 0.55}
    nm = personalized_noise_map(mi_aucs)
    assert nm["2"] > nm["1"] > nm["3"]


def test_personalized_noise_map_baseline_at_half():
    """MI AUC = 0.5 gives exactly base_sigma."""
    nm = personalized_noise_map({"1": 0.5}, base_sigma=0.05, scale=0.2)
    assert abs(nm["1"] - 0.05) < 1e-9


def test_personalized_noise_map_formula():
    base, scale = 0.05, 0.2
    mi = 0.7
    nm = personalized_noise_map({"v": mi}, base_sigma=base, scale=scale)
    expected = base + scale * (mi - 0.5)
    assert abs(nm["v"] - expected) < 1e-9


def test_personalized_noise_map_clamped_nonnegative():
    """MI AUC well below 0.5 (0.05 + 0.2*(0.2-0.5) = -0.01) must clamp to 0, not go negative."""
    nm = personalized_noise_map({"1": 0.2}, base_sigma=0.05, scale=0.2)
    assert nm["1"] == 0.0


# ---------------------------------------------------------------------------
# apply_personalized_input_noise
# ---------------------------------------------------------------------------

def test_apply_personalized_noise_total_size():
    """Output dataset has the same total number of windows as input."""
    vds = _tiny_vehicle_datasets(n_per_vehicle=40)
    nm = {"1": 0.05, "2": 0.1, "3": 0.2}
    result = apply_personalized_input_noise(vds, nm)
    total = sum(len(ds) for ds in vds.values())
    assert len(result) == total


def test_apply_personalized_noise_changes_values():
    vds = _tiny_vehicle_datasets(n_per_vehicle=20)
    nm = {"1": 0.1, "2": 0.1, "3": 0.1}
    original = torch.cat([ds.tensors[0] for ds in vds.values()])
    result = apply_personalized_input_noise(vds, nm)
    assert not torch.allclose(original, result.tensors[0])


def test_apply_personalized_noise_shape():
    vds = _tiny_vehicle_datasets(n_per_vehicle=30, w=10, f=9)
    nm = {"1": 0.05, "2": 0.1, "3": 0.2}
    result = apply_personalized_input_noise(vds, nm)
    assert result.tensors[0].shape == (90, 10, 9)


# ---------------------------------------------------------------------------
# per_vehicle_mi
# ---------------------------------------------------------------------------

def _tiny_vehicle_nonmember_datasets(n_per_vehicle=15, w=10, f=9):
    torch.manual_seed(1)
    return {
        "1": TensorDataset(torch.randn(n_per_vehicle, w, f)),
        "2": TensorDataset(torch.randn(n_per_vehicle, w, f) + 0.5),
        "3": TensorDataset(torch.randn(n_per_vehicle, w, f) - 0.5),
    }


def test_per_vehicle_mi_keys():
    vds = _tiny_vehicle_datasets(n_per_vehicle=30, w=10)
    nonmember_vds = _tiny_vehicle_nonmember_datasets(n_per_vehicle=15, w=10)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    result = per_vehicle_mi(model, vds, nonmember_vds)
    assert set(result.keys()) == {"1", "2", "3"}


def test_per_vehicle_mi_auc_range():
    vds = _tiny_vehicle_datasets(n_per_vehicle=30, w=10)
    nonmember_vds = _tiny_vehicle_nonmember_datasets(n_per_vehicle=15, w=10)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    result = per_vehicle_mi(model, vds, nonmember_vds)
    for vid, auc in result.items():
        assert 0.0 <= auc <= 1.0, f"Vehicle {vid}: MI AUC {auc} out of [0,1]"


def test_per_vehicle_mi_skips_vehicle_with_no_nonmembers():
    """A vehicle missing from (or with 0 windows in) the nonmember dict is skipped,
    not crashed on."""
    vds = _tiny_vehicle_datasets(n_per_vehicle=30, w=10)
    nonmember_vds = _tiny_vehicle_nonmember_datasets(n_per_vehicle=15, w=10)
    del nonmember_vds["2"]
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    result = per_vehicle_mi(model, vds, nonmember_vds)
    assert set(result.keys()) == {"1", "3"}
