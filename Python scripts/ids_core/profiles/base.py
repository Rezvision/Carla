"""Shared DatasetProfile dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetProfile:
    key: str
    display_name: str
    features: tuple[str, ...]
    # canonical -> possible column names in parquet/csv
    aliases: dict[str, tuple[str, ...]]
    window_size: int
    stride: int
    default_data_dir: Path
    checkpoint_dir: Path
    anomaly_safety_mult: float
    has_labels: bool
    label_col: str | None = "is_anomaly"
    failure_type_col: str | None = "failure_type"
    vehicle_id_col: str | None = "vehicle_id"
    timestamp_aliases: tuple[str, ...] = ("timestamp", "time", "ts", "datetime")
    # When True, build windows per vehicle/file (no cross-trace stitching).
    per_trace_windows: bool = True

    @property
    def n_features(self) -> int:
        return len(self.features)
