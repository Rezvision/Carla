import numpy as np
import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
from src.evaluate import (
    reconstruction_errors, make_attacks, compute_metrics, membership_inference,
    balance_members, balanced_subsample_indices, bootstrap_auc_ci,
)
from src.model import GRUAutoencoder
from src.perturbation import apply_input_noise


def _constant_model(output_val=0.0, f=9):
    """Model that always reconstructs with a fixed offset — creates known errors."""
    model = GRUAutoencoder(input_size=f, hidden_size=16, num_layers=1)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def _tiny_dataset(n=60, w=30, f=9, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(n, w, f)
    return TensorDataset(X)


def test_reconstruction_errors_shape():
    ds = _tiny_dataset(n=30)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    errors = reconstruction_errors(model, ds)
    assert errors.shape == (30,)


def test_reconstruction_errors_nonnegative():
    ds = _tiny_dataset(n=20)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    errors = reconstruction_errors(model, ds)
    assert (errors >= 0).all()


def test_make_attacks_output_shapes():
    ds = _tiny_dataset(n=40)
    attacked_ds, labels = make_attacks(ds, attack_window=3)
    assert attacked_ds.tensors[0].shape == ds.tensors[0].shape
    assert labels.shape == (40,)
    assert labels.sum() == 40  # all sequences are attacked


def test_make_attacks_modifies_sequences():
    ds = _tiny_dataset(n=20)
    attacked_ds, _ = make_attacks(ds, attack_window=3)
    assert not torch.allclose(ds.tensors[0], attacked_ds.tensors[0])


def test_make_attacks_original_unchanged():
    ds = _tiny_dataset(n=20)
    original = ds.tensors[0].clone()
    make_attacks(ds, attack_window=3)
    assert torch.allclose(ds.tensors[0], original)


def test_compute_metrics_returns_expected_keys():
    ds = _tiny_dataset(n=80, w=30)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    metrics = compute_metrics(model, ds)
    for key in ("auroc", "f1", "fpr_at_95tpr", "mean_mse_normal", "threshold"):
        assert key in metrics


def test_compute_metrics_auroc_range():
    ds = _tiny_dataset(n=80, w=30)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    metrics = compute_metrics(model, ds)
    assert 0.0 <= metrics["auroc"] <= 1.0


# ---------------------------------------------------------------------------
# Stage C: membership inference tests
# ---------------------------------------------------------------------------

def test_mi_auc_range():
    member_ds = _tiny_dataset(n=60, seed=1)
    nonmember_ds = _tiny_dataset(n=60, seed=2)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    result = membership_inference(model, member_ds, nonmember_ds)
    assert 0.0 <= result["mi_auc"] <= 1.0


def test_mi_returns_expected_keys():
    ds = _tiny_dataset(n=40)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    result = membership_inference(model, ds, ds)
    assert "mi_auc" in result
    assert "mi_accuracy" not in result


def test_mi_auc_indistinguishable():
    """Same dataset for members and non-members → identical error distributions → MI AUC = 0.5."""
    ds = _tiny_dataset(n=200, seed=1)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    result = membership_inference(model, ds, ds)
    assert abs(result["mi_auc"] - 0.5) < 0.01


def test_mi_auc_higher_with_clean_than_noisy_members_for_overfit_model():
    """
    Using noisy data as MI members suppresses MI-AUC and breaks cross-family
    comparability. Verify: for a model overfit on clean data, MI-AUC(clean members)
    > MI-AUC(noisy members). This is the invariant that compare._eval must enforce.
    """
    torch.manual_seed(42)
    n, w, f = 32, 10, 9
    clean_ds = _tiny_dataset(n=n, w=w, f=f, seed=10)
    # Large noise (σ=5.0) makes noisy inputs very different from what the model saw
    noisy_ds = apply_input_noise(clean_ds, sigma=5.0)
    nonmember_ds = _tiny_dataset(n=n, w=w, f=f, seed=99)

    # Noisy and clean tensors must genuinely differ
    assert not torch.allclose(clean_ds.tensors[0], noisy_ds.tensors[0]), \
        "Noise must change values"

    # Overfit on clean data
    model = GRUAutoencoder(input_size=f, hidden_size=32, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    loader = DataLoader(clean_ds, batch_size=n, shuffle=False)
    model.train()
    for _ in range(200):
        for (batch,) in loader:
            optimizer.zero_grad()
            criterion(model(batch), batch).backward()
            optimizer.step()

    mi_clean = membership_inference(model, clean_ds, nonmember_ds)["mi_auc"]
    mi_noisy = membership_inference(model, noisy_ds, nonmember_ds)["mi_auc"]

    assert mi_clean > mi_noisy, (
        f"MI-AUC with clean members ({mi_clean:.3f}) should exceed MI-AUC with "
        f"noisy members ({mi_noisy:.3f}). Passing noisy data as members biases the "
        f"privacy measurement — compare._eval must always use clean train_ds."
    )


def test_mi_auc_overtrained_model():
    """A heavily overfit model reconstructs training members better → MI AUC > 0.55."""
    torch.manual_seed(42)
    n, w, f = 32, 10, 9
    member_ds = _tiny_dataset(n=n, w=w, f=f, seed=10)
    nonmember_ds = _tiny_dataset(n=n, w=w, f=f, seed=99)

    model = GRUAutoencoder(input_size=f, hidden_size=32, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    loader = torch.utils.data.DataLoader(member_ds, batch_size=n, shuffle=False)

    # Overfit hard: 200 epochs, no early stopping
    model.train()
    for _ in range(200):
        for (batch,) in loader:
            optimizer.zero_grad()
            criterion(model(batch), batch).backward()
            optimizer.step()

    result = membership_inference(model, member_ds, nonmember_ds)
    assert result["mi_auc"] > 0.55, f"Expected MI AUC > 0.55 for overfit model, got {result['mi_auc']:.3f}"


# ---------------------------------------------------------------------------
# Balancing helpers (balance_members / balanced_subsample_indices)
# ---------------------------------------------------------------------------

def test_balanced_subsample_indices_deterministic():
    idx1 = balanced_subsample_indices(100, 20, seed=42)
    idx2 = balanced_subsample_indices(100, 20, seed=42)
    assert len(idx1) == 20
    assert np.array_equal(idx1, idx2)


def test_balanced_subsample_indices_noop_when_fewer_than_target():
    idx = balanced_subsample_indices(10, 50, seed=42)
    assert np.array_equal(idx, np.arange(10))


def test_balance_members_deterministic_and_sized():
    members = _tiny_dataset(n=100, seed=3)
    b1 = balance_members(members, n_nonmembers=20, seed=42)
    b2 = balance_members(members, n_nonmembers=20, seed=42)
    assert len(b1) == 20
    assert torch.equal(b1.tensors[0], b2.tensors[0])


def test_balance_members_noop_when_fewer_than_target():
    members = _tiny_dataset(n=10, seed=3)
    b = balance_members(members, n_nonmembers=50, seed=42)
    assert len(b) == 10


# ---------------------------------------------------------------------------
# bootstrap_auc_ci (cluster/block bootstrap)
# ---------------------------------------------------------------------------

def test_bootstrap_auc_ci_contains_point_estimate():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n_groups, per_group = 20, 10
    groups = np.repeat(np.arange(n_groups), per_group)
    labels = np.tile([0] * (per_group // 2) + [1] * (per_group // 2), n_groups)
    scores = labels + rng.normal(0, 0.5, size=len(labels))

    point = roc_auc_score(labels, scores)
    lo, hi = bootstrap_auc_ci(scores, labels, groups, n_boot=200, seed=1)
    assert lo <= point <= hi


def test_bootstrap_auc_ci_deterministic():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(15), 8)
    labels = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 15)
    scores = labels + rng.normal(0, 0.5, size=len(labels))

    ci1 = bootstrap_auc_ci(scores, labels, groups, n_boot=200, seed=7)
    ci2 = bootstrap_auc_ci(scores, labels, groups, n_boot=200, seed=7)
    assert ci1 == ci2


def test_bootstrap_auc_ci_single_group_no_crash():
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    labels = np.array([0, 1, 0, 1])
    groups = np.zeros(4, dtype=int)  # only one distinct group
    lo, hi = bootstrap_auc_ci(scores, labels, groups)
    assert lo == hi
    assert not np.isnan(lo)


# ---------------------------------------------------------------------------
# compute_metrics / membership_inference optional groups -> CI
# ---------------------------------------------------------------------------

def test_compute_metrics_with_groups_adds_ci_keys():
    ds = _tiny_dataset(n=80, w=30)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    groups = np.arange(80) % 8  # 8 pseudo-sessions
    metrics = compute_metrics(model, ds, groups=groups)
    assert "auroc_lo" in metrics and "auroc_hi" in metrics
    assert metrics["auroc_lo"] <= metrics["auroc"] <= metrics["auroc_hi"]


def test_membership_inference_with_groups_adds_ci_keys():
    member_ds = _tiny_dataset(n=60, seed=1)
    nonmember_ds = _tiny_dataset(n=60, seed=2)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    member_groups = np.arange(60) % 6
    nonmember_groups = np.arange(60) % 6
    result = membership_inference(
        model, member_ds, nonmember_ds,
        member_groups=member_groups, nonmember_groups=nonmember_groups,
    )
    assert "mi_auc_lo" in result and "mi_auc_hi" in result
    assert result["mi_auc_lo"] <= result["mi_auc"] <= result["mi_auc_hi"]
