import math
import torch
import pytest
from torch.utils.data import TensorDataset

from src.perturbation import (
    gaussian_epsilon,
    analytic_gaussian_epsilon,
    worst_case_sigma,
    apply_input_noise,
    apply_output_noise,
    output_noise_epsilon,
    apply_output_noise_last_layer,
    output_last_layer_epsilon,
)
from src.model import GRUAutoencoder


def _tiny_dataset(n=50, w=10, f=9, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, w, f)
    return TensorDataset(X)


def test_gaussian_epsilon_decreases_with_sigma():
    eps_small = gaussian_epsilon(0.1)
    eps_large = gaussian_epsilon(1.0)
    assert eps_small > eps_large


def test_gaussian_epsilon_zero_sigma():
    assert gaussian_epsilon(0) == float("inf")


def test_gaussian_epsilon_formula():
    sigma, sensitivity, delta = 1.0, 1.0, 1e-5
    expected = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / sigma
    assert abs(gaussian_epsilon(sigma, sensitivity, delta) - expected) < 1e-10


def test_input_noise_changes_values():
    ds = _tiny_dataset()
    noisy_ds = apply_input_noise(ds, sigma=0.5)
    assert not torch.allclose(ds.tensors[0], noisy_ds.tensors[0])


def test_input_noise_same_shape():
    ds = _tiny_dataset()
    noisy_ds = apply_input_noise(ds, sigma=0.1)
    assert noisy_ds.tensors[0].shape == ds.tensors[0].shape


def test_input_noise_original_unchanged():
    ds = _tiny_dataset()
    original = ds.tensors[0].clone()
    apply_input_noise(ds, sigma=0.5)
    assert torch.allclose(ds.tensors[0], original)


def test_output_noise_changes_params():
    model = GRUAutoencoder(hidden_size=16, num_layers=1)
    noisy = apply_output_noise(model, sigma=0.1)
    orig_params = list(model.parameters())
    noisy_params = list(noisy.parameters())
    assert any(not torch.allclose(o, n) for o, n in zip(orig_params, noisy_params))


def test_output_noise_original_unchanged():
    model = GRUAutoencoder(hidden_size=16, num_layers=1)
    orig = {k: v.clone() for k, v in model.state_dict().items()}
    apply_output_noise(model, sigma=0.5)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, orig[k])


def test_output_epsilon_positive():
    model = GRUAutoencoder(hidden_size=16, num_layers=1)
    eps = output_noise_epsilon(model, sigma=0.01)
    assert eps > 0
    assert math.isfinite(eps)


# ---------------------------------------------------------------------------
# Stage A new tests
# ---------------------------------------------------------------------------

def test_input_noise_clips_range():
    """Clipping to [-3,3] happens before noise; with sigma=0 result must stay in range."""
    torch.manual_seed(0)
    X = torch.randn(30, 10, 9) * 10  # values well outside [-3, 3]
    ds = TensorDataset(X)
    clipped_ds = apply_input_noise(ds, sigma=0.0, clip_range=3.0)
    assert clipped_ds.tensors[0].abs().max().item() <= 3.0 + 1e-5


def test_input_noise_per_feature_sigma():
    """With zero-input data, each feature's empirical std should match its assigned σ."""
    n, w, f = 8000, 10, 9
    X = torch.zeros(n, w, f)
    ds = TensorDataset(X)
    sigmas = [0.01 * (i + 1) for i in range(f)]
    noisy_ds = apply_input_noise(ds, sigma=sigmas, clip_range=10.0)
    for j, expected in enumerate(sigmas):
        actual = noisy_ds.tensors[0][:, :, j].std().item()
        assert abs(actual - expected) < 0.005, f"Feature {j}: expected σ={expected:.3f}, got {actual:.3f}"


def test_input_noise_per_feature_wrong_length():
    ds = _tiny_dataset(f=9)
    with pytest.raises(ValueError, match="length 9"):
        apply_input_noise(ds, sigma=[0.1, 0.2])  # wrong length


def test_output_last_layer_only_changes():
    """output_layer params change; encoder and decoder params are bitwise identical."""
    model = GRUAutoencoder(hidden_size=16, num_layers=1)
    noisy = apply_output_noise_last_layer(model, sigma=0.5)

    for p_orig, p_noisy in zip(model.output_layer.parameters(),
                                noisy.output_layer.parameters()):
        assert not torch.allclose(p_orig, p_noisy), "output_layer should have changed"

    for p_orig, p_noisy in zip(model.encoder.parameters(), noisy.encoder.parameters()):
        assert torch.allclose(p_orig, p_noisy), "encoder should be unchanged"

    for p_orig, p_noisy in zip(model.decoder.parameters(), noisy.decoder.parameters()):
        assert torch.allclose(p_orig, p_noisy), "decoder should be unchanged"


def test_output_last_layer_original_unchanged():
    model = GRUAutoencoder(hidden_size=16, num_layers=1)
    orig = {k: v.clone() for k, v in model.state_dict().items()}
    apply_output_noise_last_layer(model, sigma=0.5)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, orig[k])


def test_output_last_layer_epsilon_positive():
    model = GRUAutoencoder(hidden_size=16, num_layers=1)
    eps = output_last_layer_epsilon(model, sigma=0.01)
    assert eps > 0
    assert math.isfinite(eps)


def test_output_last_layer_epsilon_less_than_full():
    """Last-layer sensitivity ≤ full-model sensitivity → last-layer ε ≤ full ε."""
    model = GRUAutoencoder(hidden_size=16, num_layers=1)
    sigma = 0.01
    eps_full = output_noise_epsilon(model, sigma=sigma)
    eps_last = output_last_layer_epsilon(model, sigma=sigma)
    assert eps_last <= eps_full


# ---------------------------------------------------------------------------
# A4: analytic Gaussian mechanism (Balle & Wang 2018) + worst_case_sigma
# ---------------------------------------------------------------------------

def test_analytic_gaussian_epsilon_same_order_as_classical_for_small_eps():
    """
    In the classical formula's OWN claimed-valid regime (ε < 1), analytic and classical
    should be within the same order of magnitude and in the safe (conservative)
    direction, analytic <= classical — the classical bound is a valid but not exact
    sufficient condition even here (Balle & Wang's whole point is that it is never
    exactly tight, just "valid" for ε < 1; it is the LARGE-ε regime below where it
    breaks down qualitatively, not merely loses precision).
    """
    sigma, sensitivity, delta = 1000.0, 1.0, 1e-8
    classical = gaussian_epsilon(sigma, sensitivity, delta)
    analytic = analytic_gaussian_epsilon(sigma, sensitivity, delta)
    assert classical < 1.0, "sanity check: this case should be in the claimed-valid regime"
    ratio = analytic / classical
    assert 0.3 < ratio <= 1.0, f"expected same order of magnitude, got ratio={ratio}"


def test_analytic_gaussian_epsilon_diverges_from_classical_for_large_eps():
    """
    In the ε ≈ 10¹–10⁵ regime this project's input/output sweeps actually produce, the
    classical formula is not merely loose but qualitatively wrong: it under-counts the
    true privacy loss (analytic ε >> classical ε), the OPPOSITE direction from the
    small-ε regime above.
    """
    sigma, sensitivity, delta = 0.01, 1.0, 1e-8
    classical = gaussian_epsilon(sigma, sensitivity, delta)
    analytic = analytic_gaussian_epsilon(sigma, sensitivity, delta)
    assert analytic > 5 * classical


def test_analytic_gaussian_epsilon_finite_and_monotonic_in_sigma():
    sigmas = [0.001, 0.01, 0.1, 1.0, 10.0]
    epss = [analytic_gaussian_epsilon(s, sensitivity=1.0, delta=1e-8) for s in sigmas]
    assert all(math.isfinite(e) for e in epss)
    assert all(epss[i] > epss[i + 1] for i in range(len(epss) - 1))


def test_analytic_gaussian_epsilon_zero_sigma():
    assert analytic_gaussian_epsilon(0.0) == float("inf")


def test_worst_case_sigma_per_feature_uses_min():
    assert worst_case_sigma([0.05, 0.1, 0.3]) == 0.05


def test_worst_case_sigma_scalar_passthrough():
    assert worst_case_sigma(0.2) == 0.2
