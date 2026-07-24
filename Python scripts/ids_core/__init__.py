"""
ids_core — shared IDS training abstraction for CARLA and Kaggle datasets.

Dataset-specific knobs live in ``profiles``. Models and training do not hard-code
feature names or paths.
"""
from __future__ import annotations

from .profiles import DATASET_NAMES, DatasetProfile, get_profile
from .trainer import TrainConfig, TrainResult, train

__all__ = [
    "DATASET_NAMES",
    "DatasetProfile",
    "TrainConfig",
    "TrainResult",
    "get_profile",
    "train",
]
