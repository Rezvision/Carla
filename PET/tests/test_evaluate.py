import numpy as np
import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

import torch.nn as nn
from src.evaluate import (
    reconstruction_errors, make_attacks, compute_metrics, compute_metrics_labeled,
    membership_inference, balance_members, balanced_subsample_indices, bootstrap_auc_ci,
    bootstrap_metric_ci_iid, paired_bootstrap_diff, effective_auc,
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
    attacked_ds, labels, attack_types = make_attacks(ds, attack_window=3)
    assert attacked_ds.tensors[0].shape == ds.tensors[0].shape
    assert labels.shape == (40,)
    assert labels.sum() == 40  # all sequences are attacked
    # attack_types aligns with labels and only holds the two valid type codes.
    assert attack_types.shape == (40,)
    assert set(np.unique(attack_types)).issubset({0, 1})


def test_make_attacks_modifies_sequences():
    ds = _tiny_dataset(n=20)
    attacked_ds, _, _ = make_attacks(ds, attack_window=3)
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


def test_mi_result_keys_without_ci():
    """Without groups/iid_ci the MI result is exactly the two point estimates: raw
    mi_auc and its effective value (no CI keys, no detection metrics leaked in)."""
    ds = _tiny_dataset(n=40)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    result = membership_inference(model, ds, ds)
    assert set(result) == {"mi_auc", "mi_auc_effective"}


# ---------------------------------------------------------------------------
# PR-AUC: make_attacks(indices=...), prevalence-controlled compute_metrics,
# bootstrap_auc_ci(metric_fn=...), compute_metrics_labeled
# ---------------------------------------------------------------------------

def test_make_attacks_indices_returns_only_those_rows():
    ds = _tiny_dataset(n=40, w=30)
    idx = np.array([2, 5, 7, 11, 23])
    attacked_ds, labels, attack_types = make_attacks(ds, attack_window=3, indices=idx)
    assert attacked_ds.tensors[0].shape[0] == len(idx)
    assert labels.shape == (len(idx),)
    assert (labels == 1).all()
    assert attack_types.shape == (len(idx),)


def test_make_attacks_indices_original_unchanged():
    ds = _tiny_dataset(n=30, w=30)
    original = ds.tensors[0].clone()
    make_attacks(ds, attack_window=3, indices=np.array([1, 4, 9]))
    assert torch.allclose(ds.tensors[0], original)


def test_make_attacks_indices_none_matches_full_range():
    """indices=None must be byte-identical to passing the full arange (backward compat)."""
    ds = _tiny_dataset(n=25, w=30)
    a_none, l_none, t_none = make_attacks(ds, attack_window=3, indices=None)
    a_full, l_full, t_full = make_attacks(ds, attack_window=3, indices=np.arange(25))
    assert torch.allclose(a_none.tensors[0], a_full.tensors[0])
    assert np.array_equal(t_none, t_full)
    assert np.array_equal(l_none, l_full)


def test_compute_metrics_prevalence_adds_per_attack_type_metrics():
    """R2b: prevalence mode reports per-type recall@threshold and per-type AUROC for
    fuzzy and plateau, in [0, 1]."""
    ds = _tiny_dataset(n=300, w=30)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    m = compute_metrics(model, ds, attack_prevalence=0.2)  # higher p → both types present
    for key in ("recall_fuzzy", "recall_plateau", "auroc_fuzzy", "auroc_plateau"):
        assert key in m, f"missing {key}"
        assert 0.0 <= m[key] <= 1.0


def test_compute_metrics_balanced_has_no_per_type_metrics():
    """Per-type metrics are prevalence-mode only (they need a realistic eval set)."""
    ds = _tiny_dataset(n=80, w=30)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    m = compute_metrics(model, ds, attack_prevalence=None)
    assert not any(k.startswith(("recall_", "auroc_fuzzy", "auroc_plateau")) for k in m)


def test_compute_metrics_prevalence_pr_auc_and_prevalence():
    ds = _tiny_dataset(n=200, w=30)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    m = compute_metrics(model, ds, attack_prevalence=0.05)
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert abs(m["prevalence"] - 0.05) < 0.02
    # AUROC is still reported (prevalence-invariant) alongside PR-AUC.
    assert 0.0 <= m["auroc"] <= 1.0


def test_compute_metrics_prevalence_with_groups_pr_auc_ci():
    ds = _tiny_dataset(n=200, w=30)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    groups = np.arange(200) % 10  # 10 pseudo-sessions
    m = compute_metrics(model, ds, groups=groups, attack_prevalence=0.05)
    for key in ("auroc_lo", "auroc_hi", "pr_auc_lo", "pr_auc_hi", "prevalence"):
        assert key in m
    # Percentile CIs can occasionally exclude the point estimate, so only assert lo <= hi
    # and both in range (per the task note).
    assert m["pr_auc_lo"] <= m["pr_auc_hi"]
    assert 0.0 <= m["pr_auc_lo"] <= 1.0
    assert 0.0 <= m["pr_auc_hi"] <= 1.0


def test_compute_metrics_prevalence_none_is_nan_pr_auc_regression():
    """attack_prevalence=None → pr_auc NaN and the balanced-set metrics are unchanged
    from the original computation (regression guard)."""
    ds = _tiny_dataset(n=80, w=30)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    m = compute_metrics(model, ds, attack_prevalence=None)

    assert np.isnan(m["pr_auc"])
    # No prevalence key and no pr_auc CI on the balanced set.
    assert "prevalence" not in m
    assert "pr_auc_lo" not in m and "pr_auc_hi" not in m
    assert set(m) == {"auroc", "pr_auc", "f1", "fpr_at_95tpr", "mean_mse_normal", "threshold"}

    # Reproduce the original balanced computation and confirm the metrics match exactly.
    normal_errors = reconstruction_errors(model, ds)
    attacked_ds, attack_labels, _ = make_attacks(ds)
    attack_errors = reconstruction_errors(model, attacked_ds)
    scores = np.concatenate([normal_errors, attack_errors])
    labels = np.concatenate([np.zeros(len(normal_errors)), attack_labels])
    assert m["auroc"] == roc_auc_score(labels, scores)
    assert m["mean_mse_normal"] == float(normal_errors.mean())


def test_bootstrap_auc_ci_pr_auc_metric_fn_deterministic():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(15), 8)
    labels = np.tile([0, 0, 0, 0, 0, 0, 1, 1], 15)  # imbalanced, like a prevalence set
    scores = labels + rng.normal(0, 0.5, size=len(labels))

    ci1 = bootstrap_auc_ci(scores, labels, groups, n_boot=200, seed=7,
                           metric_fn=average_precision_score)
    ci2 = bootstrap_auc_ci(scores, labels, groups, n_boot=200, seed=7,
                           metric_fn=average_precision_score)
    assert ci1 == ci2
    lo, hi = ci1
    assert 0.0 <= lo <= hi <= 1.0


def test_compute_metrics_labeled_returns_pr_auc_and_prevalence():
    ds = _tiny_dataset(n=60, w=10, f=9)
    labels = np.array([0] * 54 + [1] * 6)  # 10% prevalence, both classes present
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    m = compute_metrics_labeled(model, ds, labels)
    assert "pr_auc" in m and 0.0 <= m["pr_auc"] <= 1.0
    assert "prevalence" in m and abs(m["prevalence"] - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# bootstrap_metric_ci_iid (i.i.d. row bootstrap — Kaggle only)
# ---------------------------------------------------------------------------

def _iid_scores_labels(n=200, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n - n // 5) + [1] * (n // 5))
    scores = labels + rng.normal(0, 0.5, size=n)
    return scores, labels


def test_bootstrap_metric_ci_iid_deterministic():
    scores, labels = _iid_scores_labels()
    ci1 = bootstrap_metric_ci_iid(scores, labels, n_boot=200, seed=7)
    ci2 = bootstrap_metric_ci_iid(scores, labels, n_boot=200, seed=7)
    assert ci1 == ci2


def test_bootstrap_metric_ci_iid_bounds_ordered_and_in_range():
    scores, labels = _iid_scores_labels()
    lo, hi = bootstrap_metric_ci_iid(scores, labels, n_boot=200, seed=1)
    assert lo <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_bootstrap_metric_ci_iid_pr_auc_metric_fn():
    """Passing average_precision_score bootstraps PR-AUC from the same row resampling."""
    scores, labels = _iid_scores_labels()
    lo, hi = bootstrap_metric_ci_iid(scores, labels, n_boot=200, seed=1,
                                     metric_fn=average_precision_score)
    assert lo <= hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_bootstrap_metric_ci_iid_different_seeds_differ():
    """The CI is a resampling estimate, not a constant — a different seed moves it."""
    scores, labels = _iid_scores_labels()
    assert (bootstrap_metric_ci_iid(scores, labels, n_boot=100, seed=1)
            != bootstrap_metric_ci_iid(scores, labels, n_boot=100, seed=2))


def test_bootstrap_metric_ci_iid_single_class_no_crash():
    """Single-class input has no defined ranking metric → NaNs, never an exception."""
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    labels = np.zeros(4, dtype=int)
    lo, hi = bootstrap_metric_ci_iid(scores, labels, n_boot=50)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_metric_ci_iid_wider_on_fewer_positives():
    """The reason this CI exists: fewer positives → a wider interval. With ~18 positives
    (the Kaggle test set) the interval must be visibly wider than with ~200."""
    rng = np.random.default_rng(3)

    def ci_width(n_pos):
        n_neg = 300
        labels = np.array([0] * n_neg + [1] * n_pos)
        scores = labels + rng.normal(0, 1.0, size=n_neg + n_pos)
        lo, hi = bootstrap_metric_ci_iid(scores, labels, n_boot=300, seed=5,
                                         metric_fn=average_precision_score)
        return hi - lo

    assert ci_width(18) > ci_width(200)


# ---------------------------------------------------------------------------
# compute_metrics_labeled(iid_ci=...) / membership_inference(iid_ci=...)
# ---------------------------------------------------------------------------

_LABELED_BASE_KEYS = {"auroc", "pr_auc", "f1", "fpr_at_95tpr", "mean_mse_normal",
                      "prevalence", "threshold"}


def test_compute_metrics_labeled_iid_ci_adds_six_keys():
    ds = _tiny_dataset(n=60, w=10, f=9)
    labels = np.array([0] * 54 + [1] * 6)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    m = compute_metrics_labeled(model, ds, labels, iid_ci=True)
    for key in ("pr_auc_lo", "pr_auc_hi", "auroc_lo", "auroc_hi"):
        assert key in m
    assert m["pr_auc_lo"] <= m["pr_auc_hi"]
    assert m["auroc_lo"] <= m["auroc_hi"]


def test_compute_metrics_labeled_iid_ci_false_output_unchanged_regression():
    """iid_ci=False must be byte-identical to the pre-CI behaviour (regression guard)."""
    ds = _tiny_dataset(n=60, w=10, f=9)
    labels = np.array([0] * 54 + [1] * 6)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)

    m_off = compute_metrics_labeled(model, ds, labels)
    assert set(m_off) == _LABELED_BASE_KEYS

    # The CI is additive only: every shared key keeps the exact same value.
    m_on = compute_metrics_labeled(model, ds, labels, iid_ci=True)
    for key in _LABELED_BASE_KEYS:
        assert m_off[key] == m_on[key]


def test_membership_inference_iid_ci_adds_mi_ci_keys():
    member_ds = _tiny_dataset(n=60, seed=1)
    nonmember_ds = _tiny_dataset(n=60, seed=2)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    result = membership_inference(model, member_ds, nonmember_ds, iid_ci=True)
    assert set(result) == {"mi_auc", "mi_auc_effective",
                           "mi_auc_lo", "mi_auc_hi", "mi_auc_eff_lo", "mi_auc_eff_hi"}
    assert result["mi_auc_lo"] <= result["mi_auc_hi"]
    assert result["mi_auc_eff_lo"] <= result["mi_auc_eff_hi"]
    # Effective CI bounds are ≥ 0.5-consistent: the effective point is in its interval.
    assert result["mi_auc_eff_lo"] <= result["mi_auc_effective"] + 1e-9


def test_mi_auc_effective_is_at_least_half_and_inverts_below_half():
    """R2a regression: mi_auc_effective = max(auc, 1-auc) ≥ 0.5, and a below-chance raw
    AUC is reported as its inverted-attack leakage, not as extra privacy."""
    ds = _tiny_dataset(n=80)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    result = membership_inference(model, ds, ds)
    assert result["mi_auc_effective"] >= 0.5 - 1e-12
    # Synthetic below-chance case: effective must fold it back above 0.5.
    from src.evaluate import effective_auc
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.8])  # members score LOW → raw AUC = 0
    assert effective_auc(labels, scores) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# paired_bootstrap_diff (paired cluster bootstrap for metric differences)
# ---------------------------------------------------------------------------

def _paired_fixture(n_groups=20, per_group=10, seed=0):
    """Two score vectors on a shared eval set: B is A plus a small positive shift."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    labels = np.tile([0] * (per_group // 2) + [1] * (per_group // 2), n_groups)
    scores_a = labels + rng.normal(0, 0.5, size=len(labels))
    return scores_a, labels, groups


def test_paired_bootstrap_diff_deterministic_with_seed():
    scores_a, labels, groups = _paired_fixture()
    scores_b = scores_a + np.random.default_rng(1).normal(0, 0.1, size=len(scores_a))
    r1 = paired_bootstrap_diff(scores_a, scores_b, labels, groups, n_boot=200, seed=7)
    r2 = paired_bootstrap_diff(scores_a, scores_b, labels, groups, n_boot=200, seed=7)
    assert r1 == r2


def test_paired_bootstrap_diff_a_vs_a_ci_contains_zero():
    """A model compared against itself has an exactly-zero difference in every
    replicate, so the interval must be degenerate at 0 (and hence contain 0)."""
    scores_a, labels, groups = _paired_fixture()
    mean_diff, lo, hi = paired_bootstrap_diff(scores_a, scores_a, labels, groups,
                                              n_boot=200, seed=7)
    assert lo <= 0.0 <= hi
    assert mean_diff == pytest.approx(0.0, abs=1e-12)


def test_paired_bootstrap_diff_detects_a_real_difference():
    """A genuinely better model gives a CI strictly above zero."""
    scores_a, labels, groups = _paired_fixture()
    # Degrade B substantially: shrink its signal toward noise.
    rng = np.random.default_rng(3)
    scores_b = 0.2 * labels + rng.normal(0, 0.5, size=len(labels))
    mean_diff, lo, hi = paired_bootstrap_diff(scores_a, scores_b, labels, groups,
                                              n_boot=300, seed=7)
    assert mean_diff > 0
    assert lo > 0, "a clearly better model should give a CI excluding zero"
    assert lo <= mean_diff <= hi


def test_paired_bootstrap_diff_shares_replicates_with_bootstrap_auc_ci():
    """Same seed ⇒ same session draws as bootstrap_auc_ci (shared conventions), so
    A-vs-A differences vanish rather than reflecting independent resampling."""
    scores_a, labels, groups = _paired_fixture()
    _, lo, hi = paired_bootstrap_diff(scores_a, scores_a, labels, groups, n_boot=100, seed=42)
    assert (lo, hi) == (0.0, 0.0)


def test_paired_bootstrap_diff_supports_effective_auc_metric():
    """The MI privacy axis uses effective_auc; the paired diff must accept it."""
    scores_a, labels, groups = _paired_fixture()
    scores_b = -scores_a  # fully inverted ⇒ same effective leakage
    mean_diff, lo, hi = paired_bootstrap_diff(scores_a, scores_b, labels, groups,
                                              metric_fn=effective_auc, n_boot=200, seed=7)
    assert mean_diff == pytest.approx(0.0, abs=1e-9)
    assert lo <= 0.0 <= hi


def test_paired_bootstrap_diff_rejects_misaligned_inputs():
    """Guard: unequal lengths mean the two models were not scored on the same set."""
    scores_a, labels, groups = _paired_fixture()
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff(scores_a, scores_a[:-5], labels, groups, n_boot=10)
