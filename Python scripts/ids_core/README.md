# ids_core — shared IDS abstraction (CARLA + Kaggle + raw CAN)

One training / model interface across datasets. Dataset differences live only
in **profiles**; model adapters never hard-code feature names. Architecture
sources live under ``models/``.

## Quick start

```bash
cd "Python scripts"

python -m experiments.train --dataset carla  --model gru  --epochs 20
python -m experiments.train --dataset kaggle --model fsmn --epochs 50

# Raw CAN (HCRL) — Chowdhury FE + GRU / VAE
python -m datasets.can.preprocess
python -m experiments.train --dataset can --model can_gru --epochs 20
python -m experiments.train --dataset can --model can_vae --epochs 50 --objective recon

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
  models/         # thin adapters (GRU / VAE / FSMN / TET / CAN-GRU / CAN-VAE)
  trainer.py      # shared train loop
  train.py        # CLI

models/           # architecture only (no train loops)
datasets/         # CSV→parquet, augmentation
experiments/      # train / compare / evaluate / artifacts
```

## Raw CAN models (`can_gru`, `can_vae`)

Shared Chowdhury-style FE (ID bit embedding, payload bytes, temporal channels)
on the `can` profile. Backbones differ:

| Model | Backbone | Detection heads |
|-------|----------|-----------------|
| `can_gru` | GRU-AE | reconstruction |
| `can_vae` | LSTM β-VAE | reconstruction, latent distance, Isolation Forest |

```bash
python -m datasets.can.preprocess                  # normal → parquet
python -m datasets.can.preprocess --include-attacks
python -m experiments.train --dataset can --model can_gru --epochs 20
python -m experiments.train --dataset can --model can_vae --epochs 50 --objective recon
```

Eval notebooks:
- `experiments/notebooks/evaluate_can_gru.ipynb`
- `experiments/notebooks/evaluate_can_vae.ipynb` (all 3 heads)

## Compatibility

| Path | Role |
|------|------|
| `mvp_v1/fed_client_jax.py` | Live edge client — **unchanged** (N_FEATURES=8) |
| `mvp_v1/train_central.py` | Thin wrapper → carla + gru → `/tmp/fed_ids_checkpoints` |
