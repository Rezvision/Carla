import torch
import pytest

from src.model import GRUAutoencoder
from src.data import FEATURES, WINDOW


B, W, F = 4, WINDOW, len(FEATURES)


def test_forward_output_shape():
    model = GRUAutoencoder(input_size=F, hidden_size=32, num_layers=1)
    x = torch.randn(B, W, F)
    out = model(x)
    assert out.shape == (B, W, F)


def test_forward_output_finite():
    model = GRUAutoencoder(input_size=F, hidden_size=32, num_layers=1)
    x = torch.randn(B, W, F)
    out = model(x)
    assert torch.isfinite(out).all()


def test_loss_finite():
    model = GRUAutoencoder(input_size=F, hidden_size=32, num_layers=1)
    x = torch.randn(B, W, F)
    loss = torch.nn.MSELoss()(model(x), x)
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_encoder_hidden_dims():
    hidden_size, num_layers = 32, 2
    model = GRUAutoencoder(input_size=F, hidden_size=hidden_size, num_layers=num_layers)
    x = torch.randn(B, W, F)
    _, h = model.encoder(x)
    assert h.shape == (num_layers, B, hidden_size)


def test_default_hyperparams():
    model = GRUAutoencoder()
    x = torch.randn(2, WINDOW, len(FEATURES))
    out = model(x)
    assert out.shape == (2, WINDOW, len(FEATURES))
