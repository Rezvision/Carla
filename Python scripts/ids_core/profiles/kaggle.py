"""Kaggle synthetic telemetry dataset profile (12 native sensors)."""
from __future__ import annotations

from pathlib import Path

from .base import DatasetProfile

_SCRIPTS = Path(__file__).resolve().parents[2]
_ROOT = _SCRIPTS.parent

KAGGLE_FEATURES = (
    "engine_rpm",
    "engine_temp_c",
    "oil_pressure_psi",
    "engine_load_percent",
    "throttle_pos_percent",
    "vibration_level",
    "brake_pedal_pos_percent",
    "brake_temp_c",
    "brake_fluid_level_psi",
    "battery_voltage_v",
    "battery_charge_percent",
    "vehicle_speed_kph",
)


def _identity_aliases(features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {f: (f,) for f in features}


KAGGLE_PROFILE = DatasetProfile(
    key="kaggle",
    display_name="Kaggle telemetry",
    features=KAGGLE_FEATURES,
    aliases=_identity_aliases(KAGGLE_FEATURES),
    window_size=20,
    stride=1,
    default_data_dir=_ROOT / "Data" / "kaggle" / "processed",
    checkpoint_dir=_SCRIPTS / "experiments" / "checkpoints" / "kaggle",
    anomaly_safety_mult=1.0,
    has_labels=True,
    label_col="is_anomaly",
    failure_type_col="failure_type",
    vehicle_id_col="vehicle_id",
    per_trace_windows=True,
)
