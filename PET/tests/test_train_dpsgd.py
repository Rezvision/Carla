import math
import pytest
import torch
from torch.utils.data import TensorDataset

from src.model import GRUAutoencoder

try:
    from src.train_dpsgd import train_dpsgd
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not OPACUS_AVAILABLE, reason="opacus not installed")


def _tiny_dataset(n=128, w=10, f=9, seed=7):
    torch.manual_seed(seed)
    return TensorDataset(torch.randn(n, w, f))


def test_dpsgd_runs_one_epoch(tmp_path):
    train_ds = _tiny_dataset(n=128)
    val_ds = _tiny_dataset(n=32, seed=8)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    save_path = str(tmp_path / "dpsgd_test.pth")
    trained_model, epsilon, epsilon_final = train_dpsgd(
        model, train_ds, val_ds, save_path=save_path,
        noise_multiplier=1.0, max_grad_norm=1.0,
        epochs=1, batch_size=32,
    )
    assert epsilon > 0
    assert math.isfinite(epsilon)
    assert epsilon_final >= epsilon


def test_dpsgd_epsilon_increases_with_epochs(tmp_path):
    train_ds = _tiny_dataset(n=128)
    val_ds = _tiny_dataset(n=32, seed=8)

    model1 = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    _, eps1, _ = train_dpsgd(
        model1, train_ds, val_ds,
        save_path=str(tmp_path / "e1.pth"),
        noise_multiplier=1.0, epochs=1, batch_size=64,
    )

    model2 = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    _, eps2, _ = train_dpsgd(
        model2, train_ds, val_ds,
        save_path=str(tmp_path / "e2.pth"),
        noise_multiplier=1.0, epochs=2, batch_size=64,
    )

    assert eps2 > eps1


def test_dpsgd_higher_noise_lower_epsilon(tmp_path):
    train_ds = _tiny_dataset(n=128)
    val_ds = _tiny_dataset(n=32, seed=8)

    model_low = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    _, eps_low_noise, _ = train_dpsgd(
        model_low, train_ds, val_ds,
        save_path=str(tmp_path / "low.pth"),
        noise_multiplier=0.5, epochs=1, batch_size=64,
    )

    model_high = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    _, eps_high_noise, _ = train_dpsgd(
        model_high, train_ds, val_ds,
        save_path=str(tmp_path / "high.pth"),
        noise_multiplier=4.0, epochs=1, batch_size=64,
    )

    assert eps_low_noise > eps_high_noise


def test_dpsgd_clip_sweep_finite_epsilon(tmp_path):
    """Both clip values in the Stage B grid produce finite ε after one epoch."""
    train_ds = _tiny_dataset(n=128)
    val_ds = _tiny_dataset(n=32, seed=8)

    for clip, i in [(0.5, "a"), (5.0, "b")]:
        model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
        _, eps, _ = train_dpsgd(
            model, train_ds, val_ds,
            save_path=str(tmp_path / f"clip_{i}.pth"),
            noise_multiplier=1.0, max_grad_norm=clip,
            epochs=1, batch_size=64,
        )
        assert eps > 0, f"clip={clip} gave ε={eps}"
        assert math.isfinite(eps), f"clip={clip} gave non-finite ε"


def test_dpsgd_clip_does_not_change_epsilon(tmp_path):
    """
    For fixed noise_multiplier, clip norm does NOT change ε.
    Opacus accounts ε from (noise_multiplier, sample_rate, steps) — the ratio
    σ/C = noise_multiplier is what matters, not C alone. Clip norm only affects
    utility (how much gradient signal is preserved), not the formal privacy budget.
    """
    train_ds = _tiny_dataset(n=128)
    val_ds = _tiny_dataset(n=32, seed=8)

    model_small = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    _, eps_small_clip, _ = train_dpsgd(
        model_small, train_ds, val_ds,
        save_path=str(tmp_path / "small_clip.pth"),
        noise_multiplier=1.0, max_grad_norm=0.5, epochs=1, batch_size=64,
    )

    model_large = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    _, eps_large_clip, _ = train_dpsgd(
        model_large, train_ds, val_ds,
        save_path=str(tmp_path / "large_clip.pth"),
        noise_multiplier=1.0, max_grad_norm=5.0, epochs=1, batch_size=64,
    )

    assert abs(eps_large_clip - eps_small_clip) < 1e-6


def _tiny_dataset_scaled(n, w, f, seed, scale=1.0, offset=0.0):
    torch.manual_seed(seed)
    return TensorDataset(torch.randn(n, w, f) * scale + offset)


def test_dpsgd_epsilon_at_best_checkpoint_not_final(tmp_path):
    """
    A1: train_dpsgd must report epsilon AT THE CHECKPOINTED (best-val) epoch, not the
    final epoch. The val set is rigged (shifted +2.0 off the train distribution) so val
    loss is lowest at epoch 1 and gets monotonically worse every epoch after — verified
    empirically for seed=7, noise_multiplier=1.0, device=cpu. A large patience keeps
    training running for all 4 epochs regardless. Opacus's epsilon is a deterministic,
    strictly increasing function of step count alone (not of data), so epsilon_at_best
    (epoch 1, fewer steps) must be strictly less than epsilon_final (epoch 4).
    """
    seed = 7
    torch.manual_seed(seed)
    model = GRUAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    train_ds = _tiny_dataset_scaled(128, 10, 9, seed=seed)
    val_ds = _tiny_dataset_scaled(32, 10, 9, seed=seed + 100, offset=2.0)

    _, eps_at_best, eps_final = train_dpsgd(
        model, train_ds, val_ds, save_path=str(tmp_path / "rigged.pth"),
        noise_multiplier=1.0, max_grad_norm=1.0,
        epochs=4, batch_size=32, patience=999,
        device=torch.device("cpu"), seed=seed,
    )

    assert eps_at_best < eps_final
