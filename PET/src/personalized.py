"""
Personalized DP: per-vehicle heterogeneous noise levels.

Single-client approximation of per-user DP (inspired by "Personalized DP for IoV",
2025). In a fully federated setting each vehicle client would control its own privacy
budget independently. Here we simulate heterogeneous noise by partitioning the single
training dataset by vehicle ID and applying a different σ per partition before training
one global model. See METHODS.md for the full caveat on this approximation.
"""
import torch
from torch.utils.data import TensorDataset

from src.perturbation import apply_input_noise


def personalized_noise_map(vehicle_mi_aucs, base_sigma=0.05, scale=0.2):
    """
    Assign per-vehicle noise level proportional to membership-inference vulnerability.

    sigma_v = max(0.0, base_sigma + scale * (mi_auc_v - 0.5))

    Vehicles with MI AUC near 0.5 (already empirically private) get noise close to
    base_sigma. Vehicles with higher MI AUC (leaking more membership) get higher noise.
    This uses MI vulnerability as a proxy for "higher privacy risk → stronger protection"
    — following the personalized-DP direction of Zhang et al. 2024 (personalized DP for
    IoV), simulated here on a single centralized dataset partitioned by vehicle (see
    METHODS.md §6 for the single-client caveat).

    Clamped to non-negative: MI AUC below 0.5 occurs in practice (the loss-threshold
    attack also picks up train/test distribution shift, not only memorisation — single-
    threshold loss attacks are noisy estimators, Carlini et al. 2022), and without
    clamping this linear rule can go negative, which is nonsensical as a noise std.

    Args:
        vehicle_mi_aucs: {vehicle_id: mi_auc} from membership_inference per vehicle.
        base_sigma: noise floor (σ at MI AUC = 0.5).
        scale: how steeply σ grows with MI AUC above 0.5.

    Returns:
        {vehicle_id: sigma}
    """
    return {
        # Linear rule: more leakage (MI-AUC above 0.5) → proportionally more noise.
        vid: max(0.0, base_sigma + scale * (mi_auc - 0.5))
        for vid, mi_auc in vehicle_mi_aucs.items()
    }


def apply_personalized_input_noise(vehicle_datasets, noise_map, clip_range=3.0, seed=42):
    """
    Apply per-vehicle Gaussian noise to a dict of {vehicle_id: TensorDataset}.
    Returns a single concatenated TensorDataset ready for training.

    Each vehicle's partition is passed through apply_input_noise with its own σ
    from noise_map (defaulting to the minimum σ for unknown vehicles).

    Args:
        vehicle_datasets: {vehicle_id: TensorDataset} — one entry per vehicle.
        noise_map: {vehicle_id: sigma} from personalized_noise_map.
        clip_range: passed to apply_input_noise (bounds local-DP sensitivity).
        seed: RNG seed (same seed across vehicles for reproducibility).
    """
    # Each vehicle gets its OWN σ; unknown vehicles fall back to the lowest σ in the map.
    default_sigma = min(noise_map.values())
    noisy_parts = []
    for vid in sorted(vehicle_datasets.keys()):
        ds = vehicle_datasets[vid]
        sigma = noise_map.get(vid, default_sigma)
        noisy_ds = apply_input_noise(ds, sigma=sigma, clip_range=clip_range, seed=seed)
        noisy_parts.append(noisy_ds.tensors[0])

    # Concatenate all per-vehicle noisy partitions into one set to train the global model.
    return TensorDataset(torch.cat(noisy_parts, dim=0))


def per_vehicle_mi(model, vehicle_train_datasets, vehicle_nonmember_datasets, device=None):
    """
    Compute MI AUC per vehicle, using that vehicle's OWN held-out (same-session)
    windows as non-members — not a shared cross-vehicle non-member set, which would
    confound vehicle-level distribution shift with memorisation (see
    split_sessions_for_mi in data.py). Members are subsampled to match each vehicle's
    non-member count (balanced 1:1, fixed seed) so per-vehicle MI-AUC isn't skewed by
    a member:non-member imbalance.

    Args:
        vehicle_train_datasets: {vehicle_id: TensorDataset} of that vehicle's training
            windows (members).
        vehicle_nonmember_datasets: {vehicle_id: TensorDataset} of that vehicle's
            held-out windows (non-members), from the SAME sessions.

    Returns {vehicle_id: mi_auc} — only for vehicles present in BOTH dicts (a vehicle
    whose holdout happened to yield zero windows is skipped).
    """
    from src.evaluate import membership_inference, balance_members
    result = {}
    for vid, member_ds in vehicle_train_datasets.items():
        nonmember_ds = vehicle_nonmember_datasets.get(vid)
        if nonmember_ds is None or len(nonmember_ds) == 0:
            continue
        balanced_members = balance_members(member_ds, len(nonmember_ds))
        result[vid] = membership_inference(model, balanced_members, nonmember_ds, device=device)["mi_auc"]
    return result
