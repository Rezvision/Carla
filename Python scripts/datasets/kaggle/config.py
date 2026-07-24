"""
Kaggle telemetry dataset config — independent of the CARLA 8-feature pipeline.

Do not import this into mvp_v1 / live edge clients.
"""
from __future__ import annotations

from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]  # .../Python scripts
ROOT = _SCRIPTS.parent  # .../Carla
DATA_DIR = ROOT / "Data"
DEFAULT_CSV = DATA_DIR / "kaggle" / "raw" / "kaggle_synthetic_telemetry_data_augmented.csv"
FALLBACK_CSV = DATA_DIR / "kaggle" / "raw" / "kaggle_synthetic_telemetry_data.csv"
PROCESSED_DIR = DATA_DIR / "kaggle" / "processed"
CHECKPOINT_DIR = _SCRIPTS / "experiments" / "checkpoints" / "kaggle"
FIGURES_DIR = _SCRIPTS / "experiments" / "figures" / "kaggle"

# Native Kaggle sensors for this experiment (not CARLA column names).
FEATURES = (
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
N_FEATURES = len(FEATURES)

# Short per-vehicle series → overlapping windows.
WINDOW_SIZE = 20
STRIDE = 1

META_COLS = (
    "vehicle_id",
    "timestamp",
    "failure_type",
    "is_anomaly",
    "engine_failure_imminent",
    "brake_issue_imminent",
    "battery_issue_imminent",
)

GRU_HIDDEN = 32
BATCH_SIZE = 32
ANOMALY_PERCENTILE = 99
# CARLA live client uses 0.37; that undershoots badly on this denser feature set.
ANOMALY_SAFETY_MULT = 1.0
NOISE_STD = 0.05
L2_LAMBDA = 1e-4
GRAD_CLIP_NORM = 5.0
LR = 1e-3
