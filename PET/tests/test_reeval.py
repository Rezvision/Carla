"""
Tests for the re-evaluation path (src.reeval + the loaders/helpers it shares with
src.compare). These cover the pure logic only — rebuilding the CARLA splits needs the
full parquet dataset, so the end-to-end run is exercised by `python -m src.reeval`.
"""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.compare import load_carla_model, _fresh_model
from src.reeval import _norm_scalar, _match_key, _epsilon_lookup, _merge_epsilon, _personalized_sigma


# ---------------------------------------------------------------------------
# load_carla_model — DP-SGD (Opacus DPGRU) checkpoint compatibility
# ---------------------------------------------------------------------------

def test_load_carla_model_roundtrips_plain_checkpoint(tmp_path):
    model = _fresh_model()
    path = tmp_path / "plain.pth"
    torch.save(model.state_dict(), path)
    loaded = load_carla_model(str(path), torch.device("cpu"))
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)


def test_load_carla_model_accepts_opacus_dpgru_extra_keys(tmp_path):
    """
    Regression: train_dpsgd runs ModuleValidator.fix, which swaps nn.GRU → Opacus DPGRU.
    The saved state_dict then carries the standard flat GRU params AND per-cell
    duplicates (encoder.l0.ih.weight, …). A strict load into a plain GRUAutoencoder
    raises "Unexpected key(s)" on those extras — which crashed the CARLA re-eval at the
    DP-SGD section. The loader must drop the DPGRU-only extras and load the flat set.
    """
    opacus_validators = pytest.importorskip("opacus.validators")
    dp_model = opacus_validators.ModuleValidator.fix(_fresh_model())
    state = dp_model.state_dict()

    # Sanity: this really is the problematic shape of state_dict.
    assert any(".l0.ih.weight" in k for k in state), "expected DPGRU per-cell keys"
    assert "encoder.weight_ih_l0" in state, "expected flat GRU keys too"

    path = tmp_path / "dpsgd.pth"
    torch.save(state, path)
    loaded = load_carla_model(str(path), torch.device("cpu"))  # must not raise

    # The flat params share storage with the DPGRU cells, so the loaded plain model must
    # hold exactly the checkpoint's trained values.
    for k, v in loaded.state_dict().items():
        assert torch.equal(v, state[k])


def test_load_carla_model_rejects_checkpoint_missing_parameters(tmp_path):
    """A checkpoint genuinely missing parameters must fail loudly, not load silently."""
    state = {k: v for k, v in _fresh_model().state_dict().items() if "output_layer" not in k}
    path = tmp_path / "incomplete.pth"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="missing parameters"):
        load_carla_model(str(path), torch.device("cpu"))


def test_load_carla_model_rejects_wrong_shape(tmp_path):
    """Filtering extras must not weaken shape checking."""
    state = dict(_fresh_model().state_dict())
    state["output_layer.weight"] = torch.randn(5, 64)
    path = tmp_path / "wrongshape.pth"
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        load_carla_model(str(path), torch.device("cpu"))


# ---------------------------------------------------------------------------
# ε carry-over matching
# ---------------------------------------------------------------------------

def test_norm_scalar_canonicalises_numeric_and_empty():
    # str-vs-float representations of the same number must collapse together …
    assert _norm_scalar("0.0001") == _norm_scalar(0.0001)
    assert _norm_scalar("1.0") == _norm_scalar(1.0)
    assert _norm_scalar("0.5") == _norm_scalar(0.5)
    # … non-numeric labels pass through …
    assert _norm_scalar("per-feature") == "per-feature"
    # … and every flavour of "absent" collapses to "".
    assert _norm_scalar(None) == ""
    assert _norm_scalar(float("nan")) == ""
    assert _norm_scalar("") == ""


def test_epsilon_carried_over_by_method_sigma_clip():
    """ε (which can't be recomputed for DP-SGD without retraining) must be matched back
    onto the re-evaluated rows on (method, σ, clip_norm) — including the DP-SGD rows that
    share a σ but differ by clip_norm."""
    old = pd.DataFrame({
        "Method": ["Baseline", "Input", "Output", "DP-SGD (fixed-clip baseline)",
                   "DP-SGD (fixed-clip baseline)", "Personalized"],
        "σ / noise_mult": [np.nan, "0.1", "0.0001", "1.0", "1.0", "v1:0.104 | v2:0.097"],
        "clip_norm": [np.nan, np.nan, np.nan, 0.5, 5.0, np.nan],
        "ε": [np.inf, 2135.76, 1.579e11, 0.0945, 0.1698, np.nan],
        "ε (final epoch)": [np.nan, np.nan, np.nan, 0.1698, 0.2208, np.nan],
    })
    lookup = _epsilon_lookup(old)

    # Rows as reeval rebuilds them: σ as float for Output/DP-SGD, str label for Input.
    results = [
        {"method": "Baseline", "sigma": None},
        {"method": "Input", "sigma": "0.1"},
        {"method": "Output", "sigma": 0.0001},
        {"method": "DP-SGD (fixed-clip baseline)", "sigma": 1.0, "clip_norm": 0.5},
        {"method": "DP-SGD (fixed-clip baseline)", "sigma": 1.0, "clip_norm": 5.0},
        {"method": "Personalized", "sigma": "v1:0.104 | v2:0.097"},
    ]
    _merge_epsilon(results, lookup)

    assert results[0]["epsilon"] == np.inf
    assert results[1]["epsilon"] == 2135.76
    assert results[2]["epsilon"] == 1.579e11
    # Same σ, different clip → must not collide.
    assert results[3]["epsilon"] == 0.0945 and results[3]["epsilon_final"] == 0.1698
    assert results[4]["epsilon"] == 0.1698 and results[4]["epsilon_final"] == 0.2208
    assert math.isnan(results[5]["epsilon"])


def test_merge_epsilon_unmatched_row_gets_nan():
    """An unmatched row gets NaN rather than a silently wrong/recomputed ε."""
    results = [{"method": "Input", "sigma": "0.99"}]
    _merge_epsilon(results, {})
    assert math.isnan(results[0]["epsilon"])
    assert math.isnan(results[0]["epsilon_final"])


def test_merge_epsilon_does_not_clobber_a_computed_epsilon():
    """
    The densified output sweep computes ε directly from baseline.pth (closed form, no
    training) at σ values that never appear in the old table. The carry-over must leave
    those alone — otherwise every dense σ would be overwritten with NaN.
    """
    old = pd.DataFrame({
        "Method": ["Output"], "σ / noise_mult": ["0.0001"], "clip_norm": [np.nan],
        "ε": [1.579e11], "ε (final epoch)": [np.nan],
    })
    results = [
        {"method": "Output", "sigma": 0.000176, "epsilon": 8.97e10},  # dense σ, computed
        {"method": "Output", "sigma": 0.0001, "epsilon": 1.579e11},   # also computed
    ]
    _merge_epsilon(results, _epsilon_lookup(old))
    assert results[0]["epsilon"] == 8.97e10, "computed ε for a dense σ was overwritten"
    assert results[1]["epsilon"] == 1.579e11
    assert all(math.isnan(r["epsilon_final"]) for r in results)


def test_personalized_sigma_recovered_from_old_table():
    old = pd.DataFrame({"Method": ["Baseline", "Personalized"],
                        "σ / noise_mult": [np.nan, "v1:0.104 | v2:0.097 | v3:0.056"]})
    assert _personalized_sigma(old) == "v1:0.104 | v2:0.097 | v3:0.056"


def test_personalized_sigma_falls_back_without_old_table():
    assert _personalized_sigma(pd.DataFrame()) == "personalized"
