"""Dataset profiles: CARLA simulation, Kaggle telemetry, raw CAN."""
from __future__ import annotations

from pathlib import Path

from .base import DatasetProfile
from .can import CAN_PROFILE
from .carla import CARLA_PROFILE
from .kaggle import KAGGLE_PROFILE

_SCRIPTS = Path(__file__).resolve().parents[1]
_ROOT = _SCRIPTS.parent

PROFILES: dict[str, DatasetProfile] = {
    "carla": CARLA_PROFILE,
    "kaggle": KAGGLE_PROFILE,
    "can": CAN_PROFILE,
}

DATASET_NAMES = tuple(PROFILES.keys())


def get_profile(key: str) -> DatasetProfile:
    try:
        return PROFILES[key]
    except KeyError as e:
        raise KeyError(f"Unknown dataset {key!r}. Choose from {DATASET_NAMES}") from e


__all__ = [
    "CAN_PROFILE",
    "CARLA_PROFILE",
    "DATASET_NAMES",
    "DatasetProfile",
    "KAGGLE_PROFILE",
    "PROFILES",
    "get_profile",
]
