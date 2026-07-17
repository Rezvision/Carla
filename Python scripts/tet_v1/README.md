# tet_v1 — AE + Temporal Embedding Transformer IDS (translated CAN)

Centralised implementation of the **TET** core from:

> L. Tao, Z. Xiyang, *Spatial-Temporal Cooperative In-Vehicle Network Intrusion
> Detection Method Based on Federated Learning*, IEEE Access, 2025.

The paper’s full system is **FL-TET** (federated AE + TET + supervised classifier).
This directory keeps the **AE + Temporal Embedding Transformer** architecture but
trains **centrally** with an **unsupervised reconstruction** objective so it matches
`mvp_v1` / `vae_v1` / `fsmn_ae_v1` on data, windowing, attacks, and metrics.

## Model

```
Input      :  (B, T, F) window + timestamps (B, T)
AE encode  :  Dense 16 → 4  (50% compression; paper was 12 → 6)
TET        :  project to d_model=64 + temporal embedding TE(ts)  (Eq. 6)
              → 2× Transformer blocks (4 heads, FF=128)
AE decode  :  Dense 16 → F
Loss       :  MSE reconstruction (+ optional Gaussian input noise σ=0.1)
Detection  :  per-window MSE vs θ = μ + k·σ  (k=2.5)
```

Temporal embedding replaces fixed positional encoding with the message timestamp:

```
TE(p, 2i)   = sin(ts_p / 10000^(2i/d))
TE(p, 2i+1) = cos(ts_p / 10000^(2i/d))
```

## Adaptation vs the paper

| Paper | This repo |
|-------|-----------|
| Federated FedAvg | Centralised training |
| Raw CAN ID + DATA (12-d) | 8 translated continuous signals |
| Supervised multi-class FC | Unsupervised reconstruction MSE |
| Window from Car-Hacking sequences | **window=20, stride=20** (repo standard) |
| Real CAN timestamps | Column if present, else normalised row index |

## Files

| File | Purpose |
|------|---------|
| `tet_model.py` | Flax AE+TET + reconstruction detector |
| `train_tet.py` | Centralised training + threshold calibration |
| `evaluate_tet.py` | CLI synthetic-attack evaluation |
| `evaluate_tet.ipynb` | Notebook: metrics, PR curve, streaming latency p50/p99 |

## Usage

```bash
cd "Python scripts/tet_v1"
pip install "jax[cpu]" flax optax scikit-learn pandas pyarrow matplotlib

# Train
python train_tet.py ../../Data/new_data_processed --epochs 100

# Evaluate
python evaluate_tet.py ../../Data/new_data_processed --bundle checkpoints --name tet_ae
```

Or open `evaluate_tet.ipynb` from this directory.
