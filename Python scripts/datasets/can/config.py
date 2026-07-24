"""Paths / defaults for the HCRL-style Car-Hacking CAN corpus."""
from __future__ import annotations

from pathlib import Path

from models.can_gru.features import FEATURES

_SCRIPTS = Path(__file__).resolve().parents[2]
_ROOT = _SCRIPTS.parent

CAN_DIR = _ROOT / "Data" / "CAN"
RAW_DIR = CAN_DIR / "raw"
PROCESSED_DIR = CAN_DIR / "processed"

# Preferred training source (benign only).
NORMAL_CSV = RAW_DIR / "normal_run_data" / "normal_run_data.csv"

ATTACK_CSVS = {
    "dos": RAW_DIR / "DoS_dataset.csv",
    "fuzzy": RAW_DIR / "Fuzzy_dataset.csv",
    "rpm": RAW_DIR / "RPM_dataset.csv",
    "gear": RAW_DIR / "gear_dataset.csv",
}

__all__ = [
    "ATTACK_CSVS",
    "CAN_DIR",
    "FEATURES",
    "NORMAL_CSV",
    "PROCESSED_DIR",
    "RAW_DIR",
]
