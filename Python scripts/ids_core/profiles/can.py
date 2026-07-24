"""Raw CAN (HCRL) dataset profile for ``can_gru``."""
from __future__ import annotations

from pathlib import Path

from models.can_gru.features import FEATURES

from .base import DatasetProfile

_SCRIPTS = Path(__file__).resolve().parents[2]
_ROOT = _SCRIPTS.parent


def _identity_aliases(features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {f: (f,) for f in features}


CAN_PROFILE = DatasetProfile(
    key="can",
    display_name="HCRL raw CAN",
    features=FEATURES,
    aliases=_identity_aliases(FEATURES),
    # VAE paper window = 50 for raw CAN message sequences.
    window_size=50,
    stride=50,
    default_data_dir=_ROOT / "Data" / "CAN" / "processed",
    checkpoint_dir=_SCRIPTS / "experiments" / "checkpoints" / "can",
    anomaly_safety_mult=1.0,
    has_labels=True,
    label_col="is_anomaly",
    failure_type_col=None,
    vehicle_id_col=None,
    per_trace_windows=True,
)
