import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, roc_curve


# Anomaly score = how badly the autoencoder reconstructs each window (mean MSE).
def reconstruction_errors(model, dataset, batch_size=512, device=None):
    """Return per-sequence MSE as a 1-D numpy array."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            reduce_dims = (1, 2) if batch.dim() == 3 else (1,)
            mse = ((recon - batch) ** 2).mean(dim=reduce_dims)  # (B,)
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


# The real CARLA data is all normal, so to MEASURE detection we synthesise
# anomalies (random noise bursts / frozen-signal plateaus) and label them as attacks.
def make_attacks(normal_dataset, attack_window=10, seed=42):
    """
    Build an attacked dataset from normal sequences and return labels.

    Two attack types applied to each sequence at random:
      - Fuzzy:   replace a sub-window with uniform-random values in [-3, 3]
                 (standardised space; 3-sigma covers real range).
      - Plateau: freeze one random feature at its mean (0 in standardised space)
                 for attack_window consecutive frames.

    Returns:
        attacked_dataset : TensorDataset of attacked sequences
        labels           : 1-D numpy array (1 = attack, 0 = normal)
    """
    rng = np.random.default_rng(seed)
    X = normal_dataset.tensors[0].numpy().copy()  # (N, W, F)
    N, W, F = X.shape

    attack_window = min(attack_window, W // 2)  # clamp so start range is always valid

    attacked = X.copy()
    labels = np.zeros(N, dtype=int)

    for i in range(N):
        attack_type = rng.integers(2)  # 0=fuzzy, 1=plateau
        start = rng.integers(0, W - attack_window)

        if attack_type == 0:
            # Fuzzy: random values in standardised range
            attacked[i, start:start + attack_window, :] = rng.uniform(-3, 3, (attack_window, F))
        else:
            # Plateau: one feature stuck at 0 (its standardised mean)
            feat = rng.integers(F)
            attacked[i, start:start + attack_window, feat] = 0.0

        labels[i] = 1

    attacked_tensor = torch.tensor(attacked, dtype=torch.float32)
    return TensorDataset(attacked_tensor), labels


# Pick the error cutoff that best separates normal vs attack (Youden's J on the ROC).
def find_threshold(model, val_dataset, device=None):
    """Youden's J on validation set to pick anomaly threshold."""
    errors = reconstruction_errors(model, val_dataset, device=device)
    # val set is all normal → label=0; we want the threshold from a mixed set,
    # so we mix val normals with a small synthetic attack set
    attacked_ds, attack_labels = make_attacks(val_dataset)
    attack_errors = reconstruction_errors(model, attacked_ds, device=device)

    scores = np.concatenate([errors, attack_errors])
    labels = np.concatenate([np.zeros(len(errors)), attack_labels])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    return float(thresholds[np.argmax(youden)])


def balanced_subsample_indices(n_total, n_target, seed=42):
    """
    Deterministic (seeded) indices to subsample n_total items down to n_target.
    Returns arange(n_total) unchanged if n_total <= n_target (no subsampling needed).
    """
    if n_total <= n_target:
        return np.arange(n_total)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=g)[:n_target]
    return idx.numpy()


def balance_members(member_dataset, n_nonmembers, seed=42):
    """
    Randomly subsample `member_dataset` down to `n_nonmembers` windows (fixed seed), so
    the MI attack sees a balanced 1:1 member:non-member ratio instead of a skewed one
    that biases AUROC estimation (and its bootstrap CI). No-op if member_dataset
    already has <= n_nonmembers windows.
    """
    X = member_dataset.tensors[0]
    idx = balanced_subsample_indices(len(X), n_nonmembers, seed=seed)
    return TensorDataset(X[idx])


def bootstrap_auc_ci(scores, labels, groups, n_boot=1000, seed=42, ci=0.95):
    """
    Cluster (block) bootstrap 95% CI for AUROC, resampling at the GROUP level (e.g.
    session ids), not per-window.

    Stride-1 windows are heavily autocorrelated within a session/group; an i.i.d.
    per-window bootstrap would give anticonservative (too-narrow) intervals. Resampling
    whole groups with replacement preserves that correlation structure in each
    replicate.

    Args:
        scores, labels: 1-D arrays, one entry per window (matching order).
        groups: 1-D array of group ids (e.g. session index), one per window, matching
                scores/labels order — the resampling unit.
        n_boot: number of bootstrap replicates.
        seed: RNG seed for reproducibility.
        ci: confidence level (default 95%).

    Returns:
        (lo, hi): percentile CI bounds. Falls back to (point_estimate, point_estimate)
        if there's only one distinct group (can't resample meaningfully) or if every
        replicate happens to be single-class (returns NaNs in that case instead).
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    point = roc_auc_score(labels, scores)
    if len(unique_groups) < 2:
        return point, point

    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    aucs = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled_groups])
        boot_labels = labels[idx]
        # A resample that (by chance) has only one class can't produce an AUROC; skip it.
        if len(np.unique(boot_labels)) < 2:
            continue
        aucs.append(roc_auc_score(boot_labels, scores[idx]))

    if not aucs:
        return float("nan"), float("nan")

    alpha = (1 - ci) / 2
    lo, hi = np.percentile(aucs, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def membership_inference(model, member_dataset, nonmember_dataset, device=None,
                          member_groups=None, nonmember_groups=None):
    """
    Threshold-based membership inference (MI) attack using reconstruction MSE as score.

    Intuition: a model that has trained on a sequence will reconstruct it with lower
    error than an unseen sequence. Lower MSE → more likely a member.

    MI AUC = AUROC for predicting membership (label 1 = member, score = -MSE).
    A perfectly private model gives MI AUC ≈ 0.5 (random guess).
    Higher MI AUC = weaker empirical privacy.

    Reference: loss-based MI attack variant of Shokri et al. 2017.
    Unlike ε, MI AUC is comparable across local-DP and central-DP methods.

    Args:
        member_groups, nonmember_groups: optional 1-D arrays of group (e.g. session)
            ids matching member_dataset/nonmember_dataset row order. When BOTH are
            provided, also computes a 95% cluster-bootstrap CI for MI-AUC (resampled at
            the group level) and adds mi_auc_lo/mi_auc_hi to the returned dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compare reconstruction error on members (seen in training) vs non-members (unseen).
    # If members reconstruct noticeably better, the model memorised them → privacy leak.
    member_errors = reconstruction_errors(model, member_dataset, device=device)
    nonmember_errors = reconstruction_errors(model, nonmember_dataset, device=device)

    # label 1 = member, score = -MSE (lower error → higher score → predicted member)
    scores = np.concatenate([-member_errors, -nonmember_errors])
    labels = np.concatenate([
        np.ones(len(member_errors)),
        np.zeros(len(nonmember_errors)),
    ])

    # AUROC of telling members from non-members. 0.5 = attacker no better than chance.
    mi_auc = roc_auc_score(labels, scores)

    result = {"mi_auc": float(mi_auc)}
    if member_groups is not None and nonmember_groups is not None:
        groups = np.concatenate([np.asarray(member_groups), np.asarray(nonmember_groups)])
        lo, hi = bootstrap_auc_ci(scores, labels, groups)
        result["mi_auc_lo"] = lo
        result["mi_auc_hi"] = hi
    return result


def compute_metrics(model, normal_test_dataset, val_dataset=None, device=None, groups=None):
    """
    Returns dict with AUROC, F1, FPR@95TPR, mean_mse_normal.

    Builds a balanced eval set: all normal test sequences + equal-sized attacked set.
    Threshold is chosen from val_dataset (or test if val not provided).

    Args:
        groups: optional 1-D array of group (e.g. session) ids, one per row in
            normal_test_dataset (matching normal_errors' order). When provided, also
            computes a 95% cluster-bootstrap CI for AUROC (resampled at the group
            level) and adds auroc_lo/auroc_hi to the returned dict.
    """
    normal_errors = reconstruction_errors(model, normal_test_dataset, device=device)
    attacked_ds, attack_labels = make_attacks(normal_test_dataset)
    attack_errors = reconstruction_errors(model, attacked_ds, device=device)

    scores = np.concatenate([normal_errors, attack_errors])
    labels = np.concatenate([np.zeros(len(normal_errors)), attack_labels])

    auroc = roc_auc_score(labels, scores)

    # FPR at 95% TPR
    fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr_curve, 0.95)
    fpr_at_95tpr = float(fpr_curve[min(idx, len(fpr_curve) - 1)])

    # F1 at Youden threshold (from val if available, else test)
    ref_ds = val_dataset if val_dataset is not None else normal_test_dataset
    threshold = find_threshold(model, ref_ds, device=device)
    preds = (scores >= threshold).astype(int)
    f1 = f1_score(labels, preds)

    result = {
        "auroc": auroc,
        "f1": f1,
        "fpr_at_95tpr": fpr_at_95tpr,
        "mean_mse_normal": float(normal_errors.mean()),
        "threshold": threshold,
    }
    if groups is not None:
        # make_attacks preserves window order 1:1, so the attacked half shares the
        # same group assignment as the normal half it was derived from.
        combined_groups = np.concatenate([np.asarray(groups), np.asarray(groups)])
        auroc_lo, auroc_hi = bootstrap_auc_ci(scores, labels, combined_groups)
        result["auroc_lo"] = auroc_lo
        result["auroc_hi"] = auroc_hi
    return result


def compute_metrics_labeled(model, test_dataset, labels,
                            val_dataset=None, val_labels=None, device=None):
    """
    Compute IDS detection metrics using REAL anomaly labels instead of synthetic attacks.

    For datasets where true failure labels are available (e.g. the Kaggle telemetry set).
    Anomaly score = reconstruction MSE (higher = more anomalous).

    Args:
        test_dataset : TensorDataset — may include both normal and anomaly rows
        labels       : np.ndarray of int, length N (0 = normal, 1 = anomaly)
        val_dataset  : optional TensorDataset for threshold selection (preferred)
        val_labels   : np.ndarray of int matching val_dataset (required with val_dataset)

    Returns dict with: auroc, f1, fpr_at_95tpr, mean_mse_normal, threshold

    AUROC and FPR@95TPR are always computed on test. Threshold is chosen via Youden-J
    from val_dataset (if provided) or test itself (backward-compat / optimistic fallback).
    """
    errors = reconstruction_errors(model, test_dataset, device=device)

    auroc = roc_auc_score(labels, errors)

    fpr_curve, tpr_curve, thresholds = roc_curve(labels, errors)

    idx_95 = np.searchsorted(tpr_curve, 0.95)
    fpr_at_95tpr = float(fpr_curve[min(idx_95, len(fpr_curve) - 1)])

    if val_dataset is not None and val_labels is not None:
        val_errors = reconstruction_errors(model, val_dataset, device=device)
        v_fpr, v_tpr, v_thresh = roc_curve(val_labels, val_errors)
        threshold = float(v_thresh[int(np.argmax(v_tpr - v_fpr))])
    else:
        threshold = float(thresholds[int(np.argmax(tpr_curve - fpr_curve))])

    preds = (errors >= threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)

    normal_errors = errors[labels == 0]
    return {
        "auroc": float(auroc),
        "f1": float(f1),
        "fpr_at_95tpr": fpr_at_95tpr,
        "mean_mse_normal": float(normal_errors.mean()) if len(normal_errors) > 0 else float("nan"),
        "threshold": threshold,
    }