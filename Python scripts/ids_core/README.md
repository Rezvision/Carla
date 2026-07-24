# ids_core — shared IDS abstraction (CARLA + Kaggle + raw CAN)

One training / model interface across datasets. Dataset differences live only
in **profiles**; model adapters never hard-code feature names. Architecture
sources live under ``models/``.

## Quick start

```bash
cd "Python scripts"

python -m experiments.train --dataset carla  --model gru  --epochs 20
python -m experiments.train --dataset kaggle --model fsmn --epochs 50

# Raw CAN (HCRL) — Chowdhury FE + GRU backbone
python -m datasets.can.preprocess
python -m experiments.train --dataset can --model can_gru --epochs 20

# Equivalent
python -m ids_core.train --dataset carla --model gru --epochs 20
```

Checkpoints → `experiments/checkpoints/{carla|kaggle|can}/`  
Figures → `experiments/figures/{carla|kaggle|can}/`

## Layout

```
ids_core/
  profiles/       # carla.py / kaggle.py / can.py DatasetProfile
  data.py         # unified parquet → windows
  models/         # thin adapters (GRU / VAE / FSMN / TET / CAN-GRU)
  trainer.py      # shared train loop
  train.py        # CLI

models/           # architecture only (no train loops)
datasets/         # CSV→parquet, augmentation
experiments/      # train / compare / evaluate / artifacts
```

## Raw CAN model (`can_gru`)

Separate from the decoded-telemetry ``gru``. Uses the feature engineering the
VAE paper describes for raw frames (ID bit embedding, payload bytes, temporal
channels) with the GRU-AE reconstruction backbone.

```bash
python -m datasets.can.preprocess                  # normal_run_data.csv → parquet
python -m datasets.can.preprocess --include-attacks
python -m experiments.train --dataset can --model can_gru --epochs 20
```

Eval notebook: `experiments/notebooks/evaluate_can_gru.ipynb`  
(scores real HCRL DoS / Fuzzy / RPM / gear traces against held-out normal traffic).

## Compatibility

| Path | Role |
|------|------|
| `mvp_v1/fed_client_jax.py` | Live edge client — **unchanged** (N_FEATURES=8) |
| `mvp_v1/train_central.py` | Thin wrapper → carla + gru → `/tmp/fed_ids_checkpoints` |
