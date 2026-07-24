"""CARLA simulation dataset profile (8 decoded telemetry features)."""
from __future__ import annotations

from pathlib import Path

from .base import DatasetProfile

_SCRIPTS = Path(__file__).resolve().parents[2]
_ROOT = _SCRIPTS.parent

CARLA_FEATURES = (
    "speed_kmh",
    "battery_level",
    "throttle",
    "brake",
    "steering",
    "gear",
    "location_x",
    "location_y",
)

CARLA_ALIASES: dict[str, tuple[str, ...]] = {
    "speed_kmh": ("speed_kmh", "speed", "velocity"),
    "battery_level": ("battery_level", "battery", "soc"),
    "throttle": ("throttle", "throttle_pct"),
    "brake": ("brake", "brake_pct"),
    "steering": ("steering", "steer", "steering_angle"),
    "gear": ("gear", "current_gear"),
    "location_x": ("location_x", "loc_x", "pos_x", "x"),
    "location_y": ("location_y", "loc_y", "pos_y", "y"),
}

CARLA_PROFILE = DatasetProfile(
    key="carla",
    display_name="CARLA simulation",
    features=CARLA_FEATURES,
    aliases=CARLA_ALIASES,
    window_size=20,
    stride=20,
    # Prefer processed/new (main corpus). Use processed/base for the smaller set:
    #   python -m experiments.train --dataset carla ... ../Data/carla/processed/base
    default_data_dir=_ROOT / "Data" / "carla" / "processed" / "new",
    checkpoint_dir=_SCRIPTS / "experiments" / "checkpoints" / "carla",
    anomaly_safety_mult=0.37,  # matches live fed_client_jax
    has_labels=False,
    label_col=None,
    failure_type_col=None,
    vehicle_id_col=None,  # one parquet file ≈ one trace
    per_trace_windows=True,
)
