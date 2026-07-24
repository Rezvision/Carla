#!/usr/bin/env python3
"""
Bootstrap + noise augmentation for kaggle_synthetic_telemetry_data.csv.

Resamples existing rows with replacement, adds Gaussian noise to numeric
sensors (scaled by each feature's empirical std), clamps to plausible
ranges, and writes longer per-vehicle sequences with new IDs.

The original CSV is never modified. Failures are bootstrapped separately
so anomaly labels remain available for evaluation (never mix into "clean"
training without filtering on failure_type).

Examples
--------
    # Default: keep originals + ~10k normal + ~500 anomalous synthetic rows
    python -m datasets.kaggle.augment

    python -m datasets.kaggle.augment \\
        --n-normal 15000 --n-anomaly 800 --noise-scale 0.05 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Columns that are not continuous sensors
META_COLS = {
    "vehicle_id",
    "brand",
    "timestamp",
    "failure_date",
    "failure_type",
    "engine_failure_imminent",
    "brake_issue_imminent",
    "battery_issue_imminent",
}

# Soft clamps derived from the source CSV (+ small margin). Binary/int fields
# are handled specially after noise.
CLAMP = {
    "odometer_reading": (0.0, 150_000.0),
    "engine_temp_c": (70.0, 130.0),
    "engine_rpm": (600.0, 4500.0),
    "oil_pressure_psi": (0.0, 100.0),
    "coolant_temp_c": (70.0, 110.0),
    "fuel_level_percent": (0.0, 100.0),
    "fuel_consumption_lph": (0.0, 20.0),
    "engine_load_percent": (0.0, 100.0),
    "throttle_pos_percent": (0.0, 100.0),
    "air_flow_rate_gps": (0.0, 120.0),
    "exhaust_gas_temp_c": (-100.0, 1100.0),
    "vibration_level": (0.0, 10.0),
    "engine_hours": (0.0, 6000.0),
    "brake_fluid_level_psi": (300.0, 1500.0),
    "brake_pad_wear_mm": (0.0, 12.0),
    "brake_temp_c": (-100.0, 450.0),
    "brake_pedal_pos_percent": (0.0, 100.0),
    "wheel_speed_fl_kph": (-60.0, 200.0),
    "wheel_speed_fr_kph": (-60.0, 200.0),
    "wheel_speed_rl_kph": (-60.0, 200.0),
    "wheel_speed_rr_kph": (-60.0, 200.0),
    "battery_voltage_v": (8.0, 16.0),
    "battery_current_a": (-100.0, 100.0),
    "battery_temp_c": (0.0, 50.0),
    "alternator_output_v": (12.0, 16.0),
    "battery_charge_percent": (0.0, 100.0),
    "battery_health_percent": (80.0, 100.0),
    "vehicle_speed_kph": (0.0, 200.0),
    "ambient_temp_c": (-30.0, 70.0),
    "humidity_percent": (0.0, 100.0),
    "gps_latitude": (27.0, 36.0),
    "gps_longitude": (74.0, 86.0),
}

BINARY_COLS = {"abs_fault_indicator"}
INT_LABEL_COLS = {
    "engine_failure_imminent",
    "brake_issue_imminent",
    "battery_issue_imminent",
}

BRANDS = [
    "Audi",
    "BMW",
    "Chevrolet",
    "Ford",
    "Honda",
    "Hyundai",
    "Kia",
    "Mercedes-Benz",
    "Nissan",
    "Toyota",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]  # .../Carla
    raw = root / "Data" / "kaggle" / "raw"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input",
        type=Path,
        default=raw / "kaggle_synthetic_telemetry_data.csv",
        help="Source CSV (left untouched)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=raw / "kaggle_synthetic_telemetry_data_augmented.csv",
        help="Augmented CSV path",
    )
    ap.add_argument("--n-normal", type=int, default=10_000, help="Synthetic normal rows to add")
    ap.add_argument("--n-anomaly", type=int, default=500, help="Synthetic anomalous rows to add")
    ap.add_argument(
        "--noise-scale",
        type=float,
        default=0.05,
        help="Gaussian noise std = noise_scale * feature_std",
    )
    ap.add_argument(
        "--rows-per-vehicle",
        type=int,
        default=200,
        help="Target sequence length for each synthetic vehicle",
    )
    ap.add_argument(
        "--include-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend original rows before synthetics (default: true)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--gap-minutes",
        type=float,
        default=30.0,
        help="Nominal timestamp step between synthetic rows (minutes)",
    )
    return ap.parse_args()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header in {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def feature_stats(rows: list[dict[str, str]], feature_cols: list[str]) -> dict[str, float]:
    stds: dict[str, float] = {}
    for col in feature_cols:
        vals = [float(r[col]) for r in rows]
        s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        stds[col] = s if s > 1e-8 else 1.0
    return stds


def clamp(col: str, value: float) -> float:
    lo, hi = CLAMP.get(col, (None, None))
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def perturb_row(
    src: dict[str, str],
    feature_cols: list[str],
    stds: dict[str, float],
    noise_scale: float,
    rng: random.Random,
) -> dict[str, str]:
    out = dict(src)
    for col in feature_cols:
        base = float(src[col])
        if col in BINARY_COLS:
            # Rare bit flips keep class balance roughly similar
            if rng.random() < min(0.02, noise_scale):
                out[col] = "1" if int(float(src[col])) == 0 else "0"
            else:
                out[col] = str(int(float(src[col])))
            continue
        noisy = base + rng.gauss(0.0, noise_scale * stds[col])
        out[col] = f"{clamp(col, noisy):.10g}"
    return out


def assign_labels_from_failure(row: dict[str, str]) -> None:
    """Keep imminent flags consistent with failure_type for synthetics."""
    ft = row["failure_type"]
    row["engine_failure_imminent"] = "0"
    row["brake_issue_imminent"] = "0"
    row["battery_issue_imminent"] = "0"
    if ft == "No Failure":
        row["failure_date"] = "2050-01-01 00:00:00"
        return
    engine = {"Engine Overheat", "Low Oil Pressure", "Excessive Vibration"}
    brake = {"Low Brake Fluid", "Brake Pad Worn", "Brake Overheat"}
    battery = {"Battery Dead", "Battery Drain", "Low Battery Voltage"}
    if ft in engine:
        row["engine_failure_imminent"] = "1"
    if ft in brake:
        row["brake_issue_imminent"] = "1"
    if ft in battery:
        row["battery_issue_imminent"] = "1"
    # Keep source failure_date if present; otherwise mark "known"
    if not row.get("failure_date") or row["failure_date"].startswith("2050"):
        row["failure_date"] = row["timestamp"]


def chunk_into_vehicles(
    rows: list[dict[str, str]],
    rows_per_vehicle: int,
    id_start: int,
    brands: list[str],
    gap_minutes: float,
    rng: random.Random,
    start_time: datetime,
) -> list[dict[str, str]]:
    """Group flat synthetic rows into vehicle sequences with timestamps."""
    out: list[dict[str, str]] = []
    vid = id_start
    t0 = start_time
    for i in range(0, len(rows), rows_per_vehicle):
        chunk = rows[i : i + rows_per_vehicle]
        if not chunk:
            break
        brand = rng.choice(brands)
        vehicle_id = f"VEH{vid:04d}"
        # small jitter on gap so sequences are not perfectly regular
        cursor = t0 + timedelta(days=rng.randint(0, 30), minutes=rng.randint(0, 120))
        for j, row in enumerate(chunk):
            row = dict(row)
            row["vehicle_id"] = vehicle_id
            row["brand"] = brand
            gap = gap_minutes * rng.uniform(0.7, 1.3)
            if j > 0:
                cursor = cursor + timedelta(minutes=gap)
            row["timestamp"] = cursor.strftime("%Y-%m-%d %H:%M:%S")
            assign_labels_from_failure(row)
            # slow drift for cumulative fields within a vehicle
            if j > 0:
                try:
                    prev_odo = float(out[-1]["odometer_reading"])
                    prev_hrs = float(out[-1]["engine_hours"])
                    row["odometer_reading"] = f"{clamp('odometer_reading', prev_odo + abs(rng.gauss(8.0, 3.0))):.10g}"
                    row["engine_hours"] = f"{clamp('engine_hours', prev_hrs + abs(rng.gauss(0.4, 0.15))):.10g}"
                except (KeyError, ValueError):
                    pass
            out.append(row)
        vid += 1
    return out


def bootstrap_pool(
    pool: list[dict[str, str]],
    n: int,
    feature_cols: list[str],
    stds: dict[str, float],
    noise_scale: float,
    rng: random.Random,
) -> list[dict[str, str]]:
    if not pool:
        return []
    out = []
    for _ in range(n):
        src = rng.choice(pool)
        out.append(perturb_row(src, feature_cols, stds, noise_scale, rng))
    return out


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    fieldnames, rows = load_csv(args.input)
    feature_cols = [c for c in fieldnames if c not in META_COLS and c not in BINARY_COLS]
    # abs_fault is numeric-ish; include in perturbation path
    feature_cols_with_binary = feature_cols + [c for c in fieldnames if c in BINARY_COLS]

    normal = [r for r in rows if r["failure_type"] == "No Failure"]
    anomaly = [r for r in rows if r["failure_type"] != "No Failure"]
    if not normal:
        raise SystemExit("No normal rows to bootstrap from")

    stds = feature_stats(normal, [c for c in feature_cols_with_binary if c not in BINARY_COLS])
    # binary std unused; ensure key exists
    for c in BINARY_COLS:
        stds[c] = 1.0

    print(f"[Augment] source rows: {len(rows)} ({len(normal)} normal, {len(anomaly)} anomaly)")
    print(f"[Augment] synthesizing {args.n_normal} normal + {args.n_anomaly} anomaly "
          f"(noise_scale={args.noise_scale}, seed={args.seed})")

    synth_normal = bootstrap_pool(
        normal, args.n_normal, feature_cols_with_binary, stds, args.noise_scale, rng
    )
    # If anomaly pool is tiny, oversample with noise (still labeled by source type)
    anom_pool = anomaly if anomaly else normal  # fallback should not happen
    if not anomaly:
        print("[Augment] WARNING: no anomaly rows in source; skipping anomaly synthesis")
        synth_anom: list[dict[str, str]] = []
    else:
        synth_anom = bootstrap_pool(
            anom_pool, args.n_anomaly, feature_cols_with_binary, stds, args.noise_scale, rng
        )

    # Original max vehicle index
    max_vid = 0
    for r in rows:
        vid = r["vehicle_id"]
        if vid.startswith("VEH"):
            try:
                max_vid = max(max_vid, int(vid[3:]))
            except ValueError:
                pass

    brands = sorted({r["brand"] for r in rows}) or BRANDS
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    synth_normal_seq = chunk_into_vehicles(
        synth_normal,
        args.rows_per_vehicle,
        id_start=max_vid + 1,
        brands=brands,
        gap_minutes=args.gap_minutes,
        rng=rng,
        start_time=start_time,
    )
    next_id = max_vid + 1 + math.ceil(len(synth_normal) / args.rows_per_vehicle)
    synth_anom_seq = chunk_into_vehicles(
        synth_anom,
        # keep anomaly vehicles shorter / denser so failures are not diluted
        rows_per_vehicle=max(20, args.rows_per_vehicle // 4),
        id_start=next_id,
        brands=brands,
        gap_minutes=args.gap_minutes,
        rng=rng,
        start_time=start_time + timedelta(days=60),
    )

    combined: list[dict[str, str]] = []
    if args.include_original:
        # Ensure originals are sorted for sequence models
        originals = sorted(rows, key=lambda r: (r["vehicle_id"], r["timestamp"]))
        combined.extend(originals)
    combined.extend(synth_normal_seq)
    combined.extend(synth_anom_seq)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in combined:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    n_fail = sum(1 for r in combined if r["failure_type"] != "No Failure")
    n_veh = len({r["vehicle_id"] for r in combined})
    print(f"[Augment] wrote {len(combined)} rows ({n_fail} anomalous) across {n_veh} vehicles")
    print(f"[Augment] output: {args.output}")
    print("[Augment] train tip: filter failure_type == 'No Failure' before fitting the AE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
