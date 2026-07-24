"""Model registry."""
from __future__ import annotations

from typing import Any

from .base import ModelSpec, ReconstructionModel
from .can_gru import CANGRUAdapter
from .can_gru import SPEC as CAN_GRU_SPEC
from .fsmn import FSMNAdapter
from .fsmn import SPEC as FSMN_SPEC
from .gru import GRUAdapter
from .gru import SPEC as GRU_SPEC
from .tet import TETAdapter
from .tet import SPEC as TET_SPEC
from .vae import VAEAdapter
from .vae import SPEC as VAE_SPEC

_SPECS = {
    GRU_SPEC.key: GRU_SPEC,
    VAE_SPEC.key: VAE_SPEC,
    FSMN_SPEC.key: FSMN_SPEC,
    TET_SPEC.key: TET_SPEC,
    CAN_GRU_SPEC.key: CAN_GRU_SPEC,
}
_BUILDERS = {
    "gru": GRUAdapter,
    "vae": VAEAdapter,
    "fsmn": FSMNAdapter,
    "tet": TETAdapter,
    "can_gru": CANGRUAdapter,
}
MODEL_NAMES = tuple(_BUILDERS.keys())


def get_spec(key: str) -> ModelSpec:
    try:
        return _SPECS[key]
    except KeyError as e:
        raise KeyError(f"Unknown model {key!r}. Choose from {MODEL_NAMES}") from e


def build_model(key: str, **kwargs: Any) -> ReconstructionModel:
    try:
        cls = _BUILDERS[key]
    except KeyError as e:
        raise KeyError(f"Unknown model {key!r}. Choose from {MODEL_NAMES}") from e
    return cls(**kwargs)  # type: ignore[return-value]
