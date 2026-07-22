import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, f1_score, roc_curve,
    average_precision_score, precision_recall_curve,
)

# Synthetic attack types produced by make_attacks (also the values in its returned
# attack_types array). Named so per-type detection metrics read clearly.
ATTACK_FUZZY = 0     # a sub-window replaced by uniform noise — a large, easy-to-spot burst
ATTACK_PLATEAU = 1   # one feature frozen at its mean — a subtle, hard-to-spot freeze
ATTACK_TYPE_NAMES = {ATTACK_FUZZY: "fuzzy", ATTACK_PLATEAU: "plateau"}


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
def make_attacks(normal_dataset, attack_window=10, seed=42, indices=None):
    """
    Build an attacked dataset from normal sequences and return labels.

    Two attack types applied to each sequence at random:
      - Fuzzy:   replace a sub-window with uniform-random values in [-3, 3]
                 (standardised space; 3-sigma covers real range).
      - Plateau: freeze one random feature at its mean (0 in standardised space)
                 for attack_window consecutive frames.

    Args:
        indices: optional 1-D int array. When given, attack ONLY those rows of
            normal_dataset and return an attacked dataset of just those rows (all
            labels 1). This is what lets compute_metrics build a realistic-prevalence
            eval set (a small attacked minority against all-normal majority) rather
            than the balanced 1:1 mix. `indices=None` keeps the original behaviour
            exactly — every window is attacked, in order — so find_threshold and the
            existing tests are unaffected.

    Returns:
        attacked_dataset : TensorDataset of attacked sequences (one per attacked row)
        labels           : 1-D numpy array, all ones (every returned row is an attack)
        attack_types     : 1-D numpy array, per-row attack type (ATTACK_FUZZY=0,
                           ATTACK_PLATEAU=1). Exposed so callers can measure detection
                           per attack type — the PR-curve cliff at recall ≈ 0.5 turned
                           out to be fuzzy (easy) vs plateau (hard), a 50/50 split.
    """
    rng = np.random.default_rng(seed)
    X = normal_dataset.tensors[0].numpy().copy()  # (N, W, F)
    N, W, F = X.shape

    attack_window = min(attack_window, W // 2)  # clamp so start range is always valid

    # None → attack every window in order (backward compatible). Otherwise attack only
    # the requested rows, in the order given, so callers can align labels/groups to
    # `indices` positionally.
    rows = np.arange(N) if indices is None else np.asarray(indices)

    attacked = X[rows].copy()
    labels = np.ones(len(rows), dtype=int)
    attack_types = np.empty(len(rows), dtype=int)

    for j, _ in enumerate(rows):
        attack_type = rng.integers(2)  # 0=fuzzy, 1=plateau
        start = rng.integers(0, W - attack_window)

        if attack_type == ATTACK_FUZZY:
            # Fuzzy: random values in standardised range
            attacked[j, start:start + attack_window, :] = rng.uniform(-3, 3, (attack_window, F))
        else:
            # Plateau: one feature stuck at 0 (its standardised mean)
            feat = rng.integers(F)
            attacked[j, start:start + attack_window, feat] = 0.0

        attack_types[j] = attack_type

    attacked_tensor = torch.tensor(attacked, dtype=torch.float32)
    return TensorDataset(attacked_tensor), labels, attack_types


# Pick the error cutoff that best separates normal vs attack (Youden's J on the ROC).
def find_threshold(model, val_dataset, device=None):
    """Youden's J on validation set to pick anomaly threshold."""
    errors = reconstruction_errors(model, val_dataset, device=device)
    # val set is all normal → label=0; we want the threshold from a mixed set,
    # so we mix val normals with a small synthetic attack set
    attacked_ds, attack_labels, _ = make_attacks(val_dataset)
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


def bootstrap_auc_ci(scores, labels, groups, n_boot=1000, seed=42, ci=0.95,
                     metric_fn=roc_auc_score):
    """
    Cluster (block) bootstrap 95% CI for a ranking metric, resampling at the GROUP
    level (e.g. session ids), not per-window.

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
        metric_fn: the metric to bootstrap, called as metric_fn(labels, scores).
                Defaults to roc_auc_score (AUROC). Pass average_precision_score to get
                a PR-AUC CI from the SAME cluster resampling — this is why PR-AUC and
                AUROC CIs on the prevalence-controlled eval set stay directly
                comparable (identical group draws, only the summary statistic differs).

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

    point = metric_fn(labels, scores)
    if len(unique_groups) < 2:
        return point, point

    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    aucs = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled_groups])
        boot_labels = labels[idx]
        # A resample that (by chance) has only one class can't produce a ranking metric; skip it.
        if len(np.unique(boot_labels)) < 2:
            continue
        aucs.append(metric_fn(boot_labels, scores[idx]))

    if not aucs:
        return float("nan"), float("nan")

    alpha = (1 - ci) / 2
    lo, hi = np.percentile(aucs, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def paired_bootstrap_diff(scores_a, scores_b, labels, groups, metric_fn=roc_auc_score,
                          n_boot=1000, seed=42, ci=0.95):
    """
    PAIRED cluster-bootstrap CI for the difference metric(A) − metric(B).

    Every method here is evaluated on the SAME sessions, so the two models' scores are
    positively correlated. Comparing their separate CIs for overlap therefore throws away
    the pairing and is badly under-powered: two intervals can overlap substantially while
    the paired difference is unambiguously non-zero. Instead, each replicate resamples
    sessions ONCE, scores BOTH models on that same replicate, and takes the difference;
    the CI is the percentile interval of those differences.

    scores_a/scores_b must be aligned row-for-row with `labels` and `groups` (i.e. both
    models scored on the identical eval set — same attacked subset, same order).

    Uses the same rng construction and group-resampling as bootstrap_auc_ci, so with a
    shared seed the replicates are the same session draws.

    Returns:
        (mean_diff, lo, hi): mean of the bootstrap differences and its percentile CI.
        Falls back to the observed difference for all three when there is only one
        distinct group; (nan, nan, nan) if every replicate is single-class.
    """
    rng = np.random.default_rng(seed)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if not (len(scores_a) == len(scores_b) == len(labels) == len(groups)):
        raise ValueError("scores_a, scores_b, labels and groups must be the same length "
                         "(both models must be scored on the identical eval set)")

    unique_groups = np.unique(groups)
    observed = float(metric_fn(labels, scores_a) - metric_fn(labels, scores_b))
    if len(unique_groups) < 2:
        return observed, observed, observed

    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    diffs = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled_groups])
        boot_labels = labels[idx]
        # A resample that (by chance) has only one class can't produce a ranking metric.
        if len(np.unique(boot_labels)) < 2:
            continue
        diffs.append(metric_fn(boot_labels, scores_a[idx]) - metric_fn(boot_labels, scores_b[idx]))

    if not diffs:
        return float("nan"), float("nan"), float("nan")

    alpha = (1 - ci) / 2
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return float(np.mean(diffs)), float(lo), float(hi)


def bootstrap_metric_ci_iid(scores, labels, n_boot=1000, seed=42, ci=0.95,
                            metric_fn=roc_auc_score):
    """
    I.I.D. row bootstrap 95% CI for a ranking metric, resampling individual ROWS with
    replacement.

    Use this ONLY where rows are exchangeable. It is the right bootstrap for the Kaggle
    tabular set — its timestamps are shuffled and it is not a usable time series, so
    kaggle_data.py treats each row as an independent sample. It is the WRONG bootstrap
    for CARLA, whose stride-1 windows are autocorrelated within a session: use
    bootstrap_auc_ci (cluster/group resampling) there instead.

    Args:
        scores, labels: 1-D arrays, one entry per row (matching order).
        n_boot: number of bootstrap replicates.
        seed: RNG seed for reproducibility.
        ci: confidence level (default 95%).
        metric_fn: the metric to bootstrap, called as metric_fn(labels, scores).
            Defaults to roc_auc_score; pass average_precision_score for a PR-AUC CI from
            the SAME row resampling, keeping the two CIs directly comparable.

    Returns:
        (lo, hi): percentile CI bounds. Returns (nan, nan) for single-class input (a
        ranking metric is undefined there, and raising would break the caller's plotting
        path) or if every replicate happens to be single-class; returns
        (point, point) when there are too few rows to resample.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    # Guard BEFORE the point estimate: metric_fn raises on single-class input.
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    point = float(metric_fn(labels, scores))
    n = len(labels)
    if n < 2:
        return point, point

    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_labels = labels[idx]
        # A resample that (by chance) has only one class can't produce a ranking metric; skip it.
        if len(np.unique(boot_labels)) < 2:
            continue
        stats.append(metric_fn(boot_labels, scores[idx]))

    if not stats:
        return float("nan"), float("nan")

    alpha = (1 - ci) / 2
    lo, hi = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def effective_auc(labels, scores):
    """
    Effective attack AUC = max(AUC, 1 − AUC).

    An MI-AUC below 0.5 is NOT extra privacy: it means the score orders members WORSE
    than non-members, so an attacker who knows the defence simply inverts the prediction
    and achieves 1 − AUC. The honest leakage floor is therefore max(AUC, 1 − AUC), which
    is ≥ 0.5 by construction (0.5 = truly no better than chance).
    """
    a = roc_auc_score(labels, scores)
    return max(a, 1.0 - a)


def membership_inference(model, member_dataset, nonmember_dataset, device=None,
                          member_groups=None, nonmember_groups=None, iid_ci=False):
    """
    Threshold-based membership inference (MI) attack using reconstruction MSE as score.

    Intuition: a model that has trained on a sequence will reconstruct it with lower
    error than an unseen sequence. Lower MSE → more likely a member.

    MI AUC = AUROC for predicting membership (label 1 = member, score = -MSE).
    A perfectly private model gives MI AUC ≈ 0.5 (random guess).

    Returns both the raw `mi_auc` and `mi_auc_effective` = max(mi_auc, 1 − mi_auc). The
    effective value is what belongs on a privacy axis: a raw AUC below 0.5 does not mean
    the model is MORE private — an attacker aware of the defence inverts the prediction
    and gets 1 − AUC. Strong output/DP noise pushes raw MI-AUC below 0.5 (the model is
    degraded, not private), which the effective value correctly reports as leakage ≥ 0.5.

    Reference: loss-based MI attack variant of Shokri et al. 2017.
    Unlike ε, MI AUC is comparable across local-DP and central-DP methods.

    Args:
        member_groups, nonmember_groups: optional 1-D arrays of group (e.g. session)
            ids matching member_dataset/nonmember_dataset row order. When BOTH are
            provided, also computes a 95% cluster-bootstrap CI (resampled at the group
            level) and adds mi_auc_lo/mi_auc_hi (raw) and mi_auc_eff_lo/mi_auc_eff_hi
            (effective, from the SAME replicates) to the returned dict.
        iid_ci: when True, add the same CI keys from an i.i.d. ROW bootstrap instead.
            Only valid where member/non-member rows are exchangeable (Kaggle tabular —
            see bootstrap_metric_ci_iid); ignored if groups are supplied, since the
            cluster bootstrap is the stricter of the two.
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

    result = {"mi_auc": float(mi_auc), "mi_auc_effective": max(float(mi_auc), 1.0 - float(mi_auc))}

    # Effective CIs come from the SAME bootstrap replicates (same seed → same group/row
    # draws), just summarised through max(auc, 1−auc), so raw and effective intervals
    # stay consistent with each other.
    groups = None
    if member_groups is not None and nonmember_groups is not None:
        groups = np.concatenate([np.asarray(member_groups), np.asarray(nonmember_groups)])
        boot = lambda fn: bootstrap_auc_ci(scores, labels, groups, metric_fn=fn)
    elif iid_ci:
        boot = lambda fn: bootstrap_metric_ci_iid(scores, labels, metric_fn=fn)
    else:
        boot = None

    if boot is not None:
        result["mi_auc_lo"], result["mi_auc_hi"] = boot(roc_auc_score)
        result["mi_auc_eff_lo"], result["mi_auc_eff_hi"] = boot(effective_auc)
    return result


def _detection_eval_set(model, normal_test_dataset, attack_prevalence=None,
                        device=None, seed=42):
    """
    Build the (normal + synthetic-attack) anomaly scores/labels used for detection metrics.

    Two regimes:
      attack_prevalence=None → the original BALANCED set: every normal window is also
        attacked once (50/50 positives). AUROC is fine here, but PRECISION — and hence
        PR-AUC — is meaningless on a balanced set because it depends on prevalence.
      attack_prevalence=p (e.g. 0.05) → a REALISTIC-prevalence set: all N normals
        (label 0) plus a seeded-random attacked subset of n_att = round(N*p/(1-p))
        windows (label 1), giving a positive fraction ≈ p. This is the only regime in
        which PR-AUC is a sensible utility metric.

    Returns:
        scores, labels : 1-D arrays (normals first, then attacks) in matching order.
        normal_errors  : per-normal-window MSE (for mean_mse_normal).
        subset_indices : the attacked rows (for group alignment), or None in the
                         balanced regime where every row is attacked.
        attack_types   : per-attacked-row type (aligned with the attack half of
                         scores/labels), for per-type detection metrics.
    """
    normal_errors = reconstruction_errors(model, normal_test_dataset, device=device)
    N = len(normal_errors)

    if attack_prevalence is None:
        attacked_ds, attack_labels, attack_types = make_attacks(normal_test_dataset)
        subset_indices = None
    else:
        p = float(attack_prevalence)
        # n_att attacks against N normals gives prevalence n_att/(N+n_att) = p.
        n_att = int(round(N * p / (1.0 - p)))
        n_att = max(1, min(n_att, N))  # keep at least one positive; never exceed N
        rng = np.random.default_rng(seed)  # fixed seed → reproducible eval set
        subset_indices = np.sort(rng.choice(N, size=n_att, replace=False))
        attacked_ds, attack_labels, attack_types = make_attacks(normal_test_dataset, indices=subset_indices)

    attack_errors = reconstruction_errors(model, attacked_ds, device=device)
    scores = np.concatenate([normal_errors, attack_errors])
    labels = np.concatenate([np.zeros(N), attack_labels])
    return scores, labels, normal_errors, subset_indices, attack_types


def compute_metrics(model, normal_test_dataset, val_dataset=None, device=None,
                    groups=None, attack_prevalence=None):
    """
    Returns dict with AUROC, PR-AUC, F1, FPR@95TPR, mean_mse_normal, threshold.

    Eval set depends on attack_prevalence (see _detection_eval_set):
      - None → balanced 50/50 mix (backward compatible). AUROC/F1/FPR are meaningful,
        but PR-AUC on a balanced set is NOT, so pr_auc is reported as NaN rather than a
        misleading value.
      - float p → realistic-prevalence mix (all normals + attacked subset). pr_auc is
        the average-precision on this set, and `prevalence` (the actual positive
        fraction) is added to the result.

    AUROC is prevalence-invariant, so it is reported in both regimes. The anomaly
    threshold (Youden's J via find_threshold) is ROC-based and prevalence-invariant, so
    it is still selected on the balanced val mix regardless of attack_prevalence.

    Args:
        groups: optional 1-D array of group (e.g. session) ids, one per row in
            normal_test_dataset (matching normal_errors' order). When provided, adds a
            95% cluster-bootstrap CI (resampled at the group level): auroc_lo/auroc_hi
            always, plus pr_auc_lo/pr_auc_hi when attack_prevalence is set.
        attack_prevalence: None for the balanced set, or a float in (0, 1) for the
            realistic-prevalence set on which PR-AUC is computed.
    """
    scores, labels, normal_errors, subset_indices, attack_types = _detection_eval_set(
        model, normal_test_dataset, attack_prevalence=attack_prevalence, device=device
    )

    auroc = roc_auc_score(labels, scores)

    # PR-AUC only makes sense on a prevalence-controlled set; NaN on the balanced one.
    pr_auc = float("nan") if attack_prevalence is None else float(
        average_precision_score(labels, scores)
    )

    # FPR at 95% TPR
    fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr_curve, 0.95)
    fpr_at_95tpr = float(fpr_curve[min(idx, len(fpr_curve) - 1)])

    # F1 at Youden threshold (from val if available, else test). find_threshold is
    # ROC-based on a balanced val mix, so it is unaffected by attack_prevalence.
    ref_ds = val_dataset if val_dataset is not None else normal_test_dataset
    threshold = find_threshold(model, ref_ds, device=device)
    preds = (scores >= threshold).astype(int)
    f1 = f1_score(labels, preds)

    result = {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "f1": f1,
        "fpr_at_95tpr": fpr_at_95tpr,
        "mean_mse_normal": float(normal_errors.mean()),
        "threshold": threshold,
    }
    if attack_prevalence is not None:
        result["prevalence"] = float(labels.mean())
        # Per-attack-type detection (prevalence mode only): the PR-curve cliff at
        # recall ≈ 0.5 is the fuzzy-vs-plateau split. For each type, recall@threshold =
        # fraction of that type's attacks flagged, and AUROC = that type's attacks vs
        # ALL normals (so both types are scored against the same negatives).
        normal_scores = scores[:len(normal_errors)]
        attack_scores = scores[len(normal_errors):]
        for t, name in ATTACK_TYPE_NAMES.items():
            mask = attack_types == t
            n_t = int(mask.sum())
            if n_t == 0:
                result[f"recall_{name}"] = float("nan")
                result[f"auroc_{name}"] = float("nan")
                continue
            result[f"recall_{name}"] = float((attack_scores[mask] >= threshold).mean())
            t_scores = np.concatenate([normal_scores, attack_scores[mask]])
            t_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(n_t)])
            result[f"auroc_{name}"] = float(roc_auc_score(t_labels, t_scores))
    if groups is not None:
        groups = np.asarray(groups)
        if subset_indices is None:
            # Balanced set: make_attacks preserves window order 1:1, so the attacked
            # half shares the same group assignment as the normal half.
            combined_groups = np.concatenate([groups, groups])
        else:
            # Prevalence-controlled set: the attacked rows are only `subset_indices`,
            # so their groups are groups[subset_indices] (NOT a second full copy).
            combined_groups = np.concatenate([groups, groups[subset_indices]])
        auroc_lo, auroc_hi = bootstrap_auc_ci(scores, labels, combined_groups)
        result["auroc_lo"] = auroc_lo
        result["auroc_hi"] = auroc_hi
        if attack_prevalence is not None:
            pr_auc_lo, pr_auc_hi = bootstrap_auc_ci(
                scores, labels, combined_groups, metric_fn=average_precision_score
            )
            result["pr_auc_lo"] = pr_auc_lo
            result["pr_auc_hi"] = pr_auc_hi
    return result


def compute_metrics_labeled(model, test_dataset, labels,
                            val_dataset=None, val_labels=None, device=None,
                            iid_ci=False):
    """
    Compute IDS detection metrics using REAL anomaly labels instead of synthetic attacks.

    For datasets where true failure labels are available (e.g. the Kaggle telemetry set).
    Anomaly score = reconstruction MSE (higher = more anomalous).

    Args:
        test_dataset : TensorDataset — may include both normal and anomaly rows
        labels       : np.ndarray of int, length N (0 = normal, 1 = anomaly)
        val_dataset  : optional TensorDataset for threshold selection (preferred)
        val_labels   : np.ndarray of int matching val_dataset (required with val_dataset)
        iid_ci       : when True, add pr_auc_lo/hi and auroc_lo/hi from an i.i.d. ROW
                       bootstrap (see bootstrap_metric_ci_iid). Only valid where rows are
                       exchangeable — true for the Kaggle tabular set, NOT for windowed
                       sequence data. Both CIs come from the same row resampling, so they
                       are directly comparable.

    Returns dict with: auroc, pr_auc, f1, fpr_at_95tpr, mean_mse_normal, prevalence,
    threshold — plus pr_auc_lo/hi and auroc_lo/hi when iid_ci=True.

    AUROC and FPR@95TPR are always computed on test. Threshold is chosen via Youden-J
    from val_dataset (if provided) or test itself (backward-compat / optimistic fallback).

    PR-AUC (average precision) is meaningful here — unlike the CARLA synthetic case —
    because these are REAL labels at their natural imbalance (~1.78% positive); the
    imbalance is the point, so no prevalence manipulation is done. `prevalence` is the
    actual positive fraction of the test labels.
    """
    labels = np.asarray(labels)
    errors = reconstruction_errors(model, test_dataset, device=device)

    auroc = roc_auc_score(labels, errors)
    pr_auc = float(average_precision_score(labels, errors))

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
    result = {
        "auroc": float(auroc),
        "pr_auc": pr_auc,
        "f1": float(f1),
        "fpr_at_95tpr": fpr_at_95tpr,
        "mean_mse_normal": float(normal_errors.mean()) if len(normal_errors) > 0 else float("nan"),
        "prevalence": float(labels.mean()),
        "threshold": threshold,
    }
    if iid_ci:
        result["pr_auc_lo"], result["pr_auc_hi"] = bootstrap_metric_ci_iid(
            errors, labels, metric_fn=average_precision_score
        )
        result["auroc_lo"], result["auroc_hi"] = bootstrap_metric_ci_iid(
            errors, labels, metric_fn=roc_auc_score
        )
    return result