import copy
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset


# ---------------------------------------------------------------------------
# Epsilon accounting (Gaussian mechanism)
# ---------------------------------------------------------------------------

def gaussian_epsilon(sigma, sensitivity=1.0, delta=1e-8):
    """
    DEPRECATED for reporting — classical Gaussian-mechanism calibration:
    ε ≈ sensitivity * sqrt(2 * ln(1.25 / δ)) / σ

    This bound is only a valid DP guarantee for ε < 1 and "cannot be extended to the
    low privacy regime" (Balle & Wang 2018) — exactly where input/output perturbation's
    ε values land in this project's sweeps (ε ≈ 10¹–10⁵). Kept only for backward
    compatibility and existing tests; use `analytic_gaussian_epsilon` for any ε that
    will be reported or plotted (see METHODS.md §1-3).

    For input perturbation, pass sensitivity=2*clip_range (not 1.0) to make the
    local-DP claim valid — the bounded range [-clip_range, +clip_range] gives an
    L∞ sensitivity of 2*clip_range per feature.
    """
    if sigma <= 0:
        return float("inf")
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / sigma


def _standard_normal_cdf(x):
    """Φ(x), the standard normal CDF, via math.erf (no scipy dependency needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _log_norm_sf(x):
    """
    log(Q(x)) where Q(x) = 1 - Φ(x) is the standard normal survival function.
    Stable for large x (this project's ε values reach ~10⁵, where Q(x) itself
    underflows to exactly 0.0 in float64 well before x does).

    x <= 5: erfc directly (no underflow risk in that range, valid for any real x).
    x > 5: standard asymptotic expansion of the Mills ratio.
    """
    if x <= 5:
        return math.log(0.5 * math.erfc(x / math.sqrt(2.0)))
    log_phi = -0.5 * x * x - 0.5 * math.log(2 * math.pi)
    series = 1 - 1 / x**2 + 3 / x**4 - 15 / x**6 + 105 / x**8
    return log_phi - math.log(x) + math.log(series)


def _analytic_gaussian_delta(epsilon, sigma, sensitivity):
    """
    δ(ε) for the analytic Gaussian mechanism (Balle & Wang 2018, Thm 8):
    δ(ε) = Φ(a-b) - exp(ε)·Φ(-a-b), with a=Δ/2σ, b=εσ/Δ.

    Computed via log-space for the second term (exp(ε)·Φ(-a-b) = exp(ε + log Q(a+b))),
    since naive exp(ε) overflows for the ε ~ 10¹-10⁵ range this project needs, even
    though the true product shrinks (log Q(a+b) is quadratically negative in ε, which
    dominates the linear +ε term).
    """
    a = sensitivity / (2.0 * sigma)
    b = epsilon * sigma / sensitivity
    term1 = math.exp(_log_norm_sf(b - a))       # Φ(a-b) = Q(b-a)
    log_term2 = epsilon + _log_norm_sf(a + b)   # log(exp(ε) · Φ(-a-b)), a+b > 0 always
    term2 = math.exp(log_term2) if log_term2 < 700 else float("inf")
    return term1 - term2


def analytic_gaussian_epsilon(sigma, sensitivity=1.0, delta=1e-8, tol=1e-9, max_iter=200):
    """
    Exact ε for the Gaussian mechanism, valid at ANY privacy regime (Balle & Wang 2018).

    Unlike `gaussian_epsilon`'s classical bound (only valid for ε < 1), this bisects the
    analytic δ(ε) curve for the given (sigma, sensitivity) — which is monotonically
    decreasing in ε — to find where it crosses the target δ. Valid for all ε > 0, which
    is what this project's input/output perturbation ε values require.
    """
    if sigma <= 0:
        return float("inf")

    lo, hi = 0.0, 1.0
    while _analytic_gaussian_delta(hi, sigma, sensitivity) > delta:
        hi *= 2
        if hi > 1e15:
            return float("inf")
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        if _analytic_gaussian_delta(mid, sigma, sensitivity) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def worst_case_sigma(sigma):
    """
    For a per-feature σ (list), return the SMALLEST value — the least-noised feature is
    the worst case for a single scalar ε. A scalar σ passes through unchanged.
    """
    return min(sigma) if isinstance(sigma, list) else sigma


# ---------------------------------------------------------------------------
# Input perturbation
# ---------------------------------------------------------------------------

# Helper: build a (1,1,F) std tensor so BOTH a scalar σ and a per-feature σ vector
# broadcast cleanly over a (batch, time, feature) input.
def _sigma_tensor(sigma, n_features, device):
    """Return a (1, 1, F) tensor of noise stds from a scalar or per-feature sequence."""
    if np.isscalar(sigma):
        return torch.full((1, 1, n_features), float(sigma), dtype=torch.float32, device=device)
    s = torch.tensor(sigma, dtype=torch.float32, device=device)
    if s.shape != (n_features,):
        raise ValueError(f"sigma vector must have length {n_features}, got {list(s.shape)}")
    return s.view(1, 1, n_features)


def apply_input_noise(dataset, sigma, clip_range=3.0, seed=42):
    """
    Add Gaussian noise to every feature value.

    sigma      : scalar std, or sequence of length F for per-feature stds.
    clip_range : standardised features are clipped to [-clip_range, +clip_range]
                 BEFORE noise is added, bounding input sensitivity to 2*clip_range
                 and making the Gaussian-mechanism ε formula valid for local DP.
    """
    rng = torch.Generator().manual_seed(seed)
    X = dataset.tensors[0]
    # A plain TensorDataset shares storage with its caller's tensor, so clone before
    # mutating in place. A lazy WindowDataset's .tensors already returns a freshly
    # materialised tensor each call, so no clone (and no aliasing risk) there.
    if isinstance(dataset, TensorDataset):
        X = X.clone()

    # Clip to bound sensitivity (prerequisite for a valid local-DP ε)
    X.clamp_(-clip_range, clip_range)

    # Draw standard-normal noise, then scale by σ (per-feature if σ is a vector).
    sigma_t = _sigma_tensor(sigma, X.shape[-1], X.device)  # (1, 1, F)
    if X.dim() == 2:
        sigma_t = sigma_t.squeeze(0)  # → (1, F), broadcasts over (N, F) tabular input
    noise = torch.zeros_like(X).normal_(generator=rng)
    noise.mul_(sigma_t)
    X.add_(noise)

    return TensorDataset(X)


# ---------------------------------------------------------------------------
# Output perturbation — full model
# ---------------------------------------------------------------------------

def apply_output_noise(model, sigma, seed=42):
    """
    Return a deep copy of the model with Gaussian noise added to every parameter.
    The original model is unchanged.
    """
    rng = torch.Generator().manual_seed(seed)
    noisy_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in noisy_model.parameters():
            # rng is CPU-only, so draw noise on CPU then move it to the param's device
            # (params may already be on cuda after training).
            noise = torch.zeros(param.shape, dtype=param.dtype).normal_(
                mean=0.0, std=sigma, generator=rng
            ).to(param.device)
            param.add_(noise)
    return noisy_model


def output_noise_epsilon(model, sigma, delta=1e-8):
    """
    ε for full-parameter output perturbation, via the analytic Gaussian mechanism
    (Balle & Wang 2018 — valid at any ε, unlike the classical bound; see METHODS.md §2).
    Sensitivity = max per-layer L2 norm (conservative global bound).
    """
    # Loose global sensitivity = the largest single-layer L2 norm. Deep nets have no
    # tight bound, which is why full-model output perturbation needs huge noise.
    sensitivity = max(param.data.norm(2).item() for param in model.parameters())
    return analytic_gaussian_epsilon(sigma, sensitivity=sensitivity, delta=delta)


# ---------------------------------------------------------------------------
# Output perturbation — last layer only (inspired by Lu et al. 2022)
# ---------------------------------------------------------------------------

def apply_output_noise_last_layer(model, sigma, seed=42):
    """
    Return a deep copy of the model with Gaussian noise added only to
    model.output_layer parameters. Encoder and decoder weights are unchanged.

    Inspired by Lu et al. 2022 — their method uses the exponential mechanism on a
    convexified loss to perturb a single output neuron. This is a simplification:
    Gaussian noise is applied to ALL output-layer parameters rather than one neuron
    via the exponential mechanism. See METHODS.md for the full framing.
    """
    rng = torch.Generator().manual_seed(seed)
    noisy_model = copy.deepcopy(model)
    with torch.no_grad():
        # Perturb ONLY the final Linear layer — the encoder/decoder GRUs are left untouched.
        for param in noisy_model.output_layer.parameters():
            # rng is CPU-only, so draw noise on CPU then move it to the param's device
            # (params may already be on cuda after training).
            noise = torch.zeros(param.shape, dtype=param.dtype).normal_(
                mean=0.0, std=sigma, generator=rng
            ).to(param.device)
            param.add_(noise)
    return noisy_model


def output_last_layer_epsilon(model, sigma, delta=1e-8):
    """
    ε for last-layer-only output perturbation, via the analytic Gaussian mechanism
    (Balle & Wang 2018; see METHODS.md §3).
    Sensitivity = max L2 norm across output_layer weight/bias tensors.
    """
    sensitivity = max(
        param.data.norm(2).item() for param in model.output_layer.parameters()
    )
    return analytic_gaussian_epsilon(sigma, sensitivity=sensitivity, delta=delta)
