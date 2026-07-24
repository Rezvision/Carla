"""
Chowdhury-style raw-CAN feature engineering (VAE paper Sec. 3.1.1).

The repository VAE *documents* this pipeline but omits it for decoded CARLA
signals.  ``can_gru`` applies it to HCRL-style frames:

  * arbitration ID → 11 binary bits mapped into a continuous space
  * 8 payload bytes normalised to [0, 1]
  * engineered temporal features (inter-arrival, per-ID interval, entropy, DLC)

Learnable ID embedding (Dense over the ID bits) lives in ``model.py``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install deps: pip install pandas") from e

ID_BITS = 11
PAYLOAD_BYTES = 8
EMBED_DIM = 8  # learned projection width in the model

ID_BIT_FEATURES = tuple(f"id_b{i}" for i in range(ID_BITS))
PAYLOAD_FEATURES = tuple(f"data_{i}" for i in range(PAYLOAD_BYTES))
TEMPORAL_FEATURES = (
    "dlc",
    "inter_arrival",
    "id_interval",
    "payload_entropy",
)

# Columns written to parquet / listed in the ``can`` DatasetProfile.
# Order matters: model treats the first ID_BITS columns as the ID bit field.
FEATURES: tuple[str, ...] = ID_BIT_FEATURES + PAYLOAD_FEATURES + TEMPORAL_FEATURES
N_RAW_FEATURES = len(FEATURES)  # 23
N_MODEL_FEATURES = EMBED_DIM + PAYLOAD_BYTES + len(TEMPORAL_FEATURES)  # 20

# Clip / normalise constants for temporal channels.
_MAX_DT = 1.0  # seconds; larger gaps saturate
_MAX_ENTROPY = np.log2(256.0)  # Shannon entropy of a byte histogram


def can_id_to_bits(can_id: int | np.ndarray, n_bits: int = ID_BITS) -> np.ndarray:
    """Map arbitration ID(s) to continuous binary features in {0, 1}^n_bits."""
    ids = np.asarray(can_id, dtype=np.uint32).reshape(-1)
    bits = ((ids[:, None] >> np.arange(n_bits, dtype=np.uint32)) & 1).astype(np.float32)
    return bits


def payload_entropy(bytes_row: np.ndarray) -> float:
    """Normalised Shannon entropy of up to 8 payload bytes ∈ [0, 1]."""
    vals = np.asarray(bytes_row, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    # Histogram over 0..255 (bytes already in 0..255 or 0..1).
    if vals.max() <= 1.0 + 1e-6:
        vals = np.clip(vals * 255.0, 0, 255)
    hist = np.bincount(vals.astype(np.int64), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    h = float(-(p * np.log2(p)).sum())
    return float(h / _MAX_ENTROPY)


def _parse_payload_field(raw) -> list[float]:
    """Parse space-separated hex payload *or* a list of byte-like values."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        vals = [int(str(x), 16) if isinstance(x, str) else int(x) for x in raw]
    else:
        s = str(raw).strip()
        if not s:
            vals = []
        else:
            vals = [int(tok, 16) for tok in s.replace(",", " ").split()]
    # Pad / truncate to 8 bytes.
    vals = (vals + [0] * PAYLOAD_BYTES)[:PAYLOAD_BYTES]
    return [v / 255.0 for v in vals]


def _detect_columns(df: pd.DataFrame) -> dict[str, str | list[str]]:
    cols = {c.lower(): c for c in df.columns}
    # Attack CSVs: timestamp,id,dlc,b0..b7,flag (no header) — caller may assign names.
    ts = next((cols[k] for k in ("timestamp", "time", "ts") if k in cols), None)
    can_id = next((cols[k] for k in ("can_id", "id", "arb_id") if k in cols), None)
    dlc = next((cols[k] for k in ("dlc", "data_length") if k in cols), None)
    flag = next((cols[k] for k in ("flag", "label", "class") if k in cols), None)

    payload_col = next((cols[k] for k in ("payload", "data", "data_field") if k in cols), None)
    byte_cols = []
    for i in range(PAYLOAD_BYTES):
        for key in (f"data_{i}", f"d{i}", f"byte{i}", f"b{i}", str(i)):
            if key in cols:
                byte_cols.append(cols[key])
                break
    if len(byte_cols) != PAYLOAD_BYTES:
        byte_cols = []

    if ts is None or can_id is None:
        raise ValueError(
            f"Need timestamp + can_id columns. Available: {list(df.columns)}"
        )
    return {
        "timestamp": ts,
        "can_id": can_id,
        "dlc": dlc,
        "flag": flag,
        "payload": payload_col,
        "bytes": byte_cols,
    }


def _parse_hcrl_line(line: str) -> dict | None:
    """
    Parse one HCRL / Car-Hacking CSV row with *variable* payload length.

    Layout is always: ``timestamp, id, dlc, <dlc bytes...>, flag``
    so DLC=2 rows have 6 fields and DLC=8 rows have 12.  A fixed-width
    ``read_csv`` shifts the flag into a data column on short rows.
    """
    parts = [p.strip() for p in line.rstrip("\n\r").split(",")]
    if len(parts) < 4:
        return None
    ts, can_id, dlc_s, flag = parts[0], parts[1], parts[2], parts[-1]
    raw_bytes = parts[3:-1]
    # Pad / truncate to 8 bytes.
    raw_bytes = (raw_bytes + ["00"] * PAYLOAD_BYTES)[:PAYLOAD_BYTES]
    row = {
        "timestamp": ts,
        "can_id": can_id,
        "dlc": dlc_s,
        "flag": flag,
    }
    for i, b in enumerate(raw_bytes):
        row[f"data_{i}"] = b
    return row


def read_raw_can_csv(path) -> pd.DataFrame:
    """
    Load HCRL-style CAN CSV (headerless).

    Supports:
      * ``timestamp,id,dlc,payload,flag``  (payload = space-separated hex)
      * ``timestamp,id,dlc,b0,...,b{dlc-1},flag`` (variable-length byte columns)
    """
    path = str(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n\r")

    parts = first.split(",")
    n = len(parts)

    if n == 5 and " " in parts[3]:
        # timestamp, id, dlc, "aa bb ..", flag  (our cleaned normal_run CSV)
        names = ["timestamp", "can_id", "dlc", "payload", "flag"]
        return pd.read_csv(path, header=None, names=names, dtype=str)

    # Variable-length attack CSVs (and any other HCRL byte-column dumps).
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = _parse_hcrl_line(line)
            if row is not None:
                rows.append(row)
    if not rows:
        # Last resort: headered CSV.
        return pd.read_csv(path)
    return pd.DataFrame(rows)


def engineer_can_frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw CAN frames to the continuous feature matrix used by ``can_gru``.

    Returns a dataframe with ``FEATURES`` plus ``timestamp`` and ``is_anomaly``.
    """
    mapping = _detect_columns(df)
    ts = pd.to_numeric(df[mapping["timestamp"]], errors="coerce").to_numpy(dtype=np.float64)
    # CAN IDs may be hex strings ("0350") or ints.
    id_raw = df[mapping["can_id"]].astype(str).str.strip()
    can_ids = id_raw.map(lambda s: int(s, 16) if s else 0).to_numpy(dtype=np.uint32)

    if mapping["dlc"] is not None:
        dlc = pd.to_numeric(df[mapping["dlc"]], errors="coerce").fillna(8).to_numpy(dtype=np.float32)
    else:
        dlc = np.full(len(df), 8.0, dtype=np.float32)

    n = len(df)
    payload = np.zeros((n, PAYLOAD_BYTES), dtype=np.float32)
    def _hex_byte(s: str) -> float:
        s = (s or "").strip()
        if not s or s.lower() in ("nan", "none", "r", "t"):
            return 0.0
        try:
            return int(s, 16) / 255.0
        except ValueError:
            return 0.0

    if mapping["bytes"]:
        for j, col in enumerate(mapping["bytes"]):
            series = df[col].astype(str)
            payload[:, j] = series.map(_hex_byte).to_numpy(dtype=np.float32)
    elif mapping["payload"] is not None:
        parsed = df[mapping["payload"]].map(_parse_payload_field)
        payload = np.asarray(parsed.tolist(), dtype=np.float32)
    else:
        raise ValueError("No payload columns found (need 'payload' or data_0..data_7)")

    # Temporal features.
    order = np.argsort(ts, kind="mergesort")
    ts_sorted = ts[order]
    ids_sorted = can_ids[order]
    payload_sorted = payload[order]
    dlc_sorted = dlc[order]

    dt = np.zeros(n, dtype=np.float32)
    dt[1:] = np.diff(ts_sorted).astype(np.float32)
    dt = np.clip(dt, 0.0, _MAX_DT)

    id_interval = np.zeros(n, dtype=np.float32)
    last_t: dict[int, float] = {}
    for i in range(n):
        cid = int(ids_sorted[i])
        t = float(ts_sorted[i])
        if cid in last_t and np.isfinite(t) and np.isfinite(last_t[cid]):
            id_interval[i] = min(_MAX_DT, max(0.0, t - last_t[cid]))
        last_t[cid] = t

    entropy = np.array(
        [payload_entropy(payload_sorted[i]) for i in range(n)], dtype=np.float32
    )

    bits = can_id_to_bits(ids_sorted)  # (n, 11)

    # Flag → is_anomaly (T/attack = 1, R/normal = 0).
    if mapping["flag"] is not None:
        flag = df[mapping["flag"]].astype(str).str.strip().str.upper().to_numpy()
        flag = flag[order]
        is_anom = np.array(
            [1 if f in ("T", "1", "ATTACK", "ANOMALY", "TRUE") else 0 for f in flag],
            dtype=np.int8,
        )
    else:
        is_anom = np.zeros(n, dtype=np.int8)

    out = {
        "timestamp": ts_sorted.astype(np.float64),
        "is_anomaly": is_anom,
        "can_id": ids_sorted.astype(np.int32),
    }
    for i in range(ID_BITS):
        out[f"id_b{i}"] = bits[:, i]
    for i in range(PAYLOAD_BYTES):
        out[f"data_{i}"] = payload_sorted[:, i]
    out["dlc"] = (dlc_sorted / 8.0).astype(np.float32)
    out["inter_arrival"] = dt
    out["id_interval"] = id_interval
    out["payload_entropy"] = entropy
    return pd.DataFrame(out)


def feature_matrix_from_engineered(df: pd.DataFrame) -> np.ndarray:
    """Stack ``FEATURES`` columns to ``(N, N_RAW_FEATURES)`` float32."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered columns: {missing}")
    return df.loc[:, list(FEATURES)].to_numpy(dtype=np.float32)


def iter_can_sources(paths: Iterable) -> list:
    return [p for p in paths]
