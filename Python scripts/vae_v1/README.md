# vae_v1 — LSTM Variational Autoencoder IDS

Implementation of the model from:

> S. Chowdhury, *Edge-Deployable Unsupervised Intrusion Detection for CAN Using
> Variational Autoencoders and Latent Space Disentanglement*, MSc thesis,
> KTH / Scania CV AB, July 2025.

This is a sibling of `mvp_v1` (which houses the GRU autoencoder). It implements the
paper's β-VAE and its three anomaly-detection heads, trained centrally on the
translated (decoded) CAN telemetry in `Data/`.

## Model (paper Sec. 3.2, Fig. 3.3)

```
Encoder :  LSTM  ->  global average pooling  ->  Dense(mu), Dense(logvar)
           logvar clipped to [-10, 5], std = softplus(logvar)
Sampling:  reparameterisation trick + multi-sample Monte-Carlo (n=5 train, n=1 infer)
Decoder :  RepeatVector  ->  LSTM  ->  TimeDistributed Dense (no output activation)
Loss    :  beta-VAE  =  MSE reconstruction  +  beta * KL divergence
```

Hyperparameters follow Table 3.1: latent dim 10, window 50, stride 20, batch 1024,
AdamW with cosine-decay LR (1e-3 → 0), 100 epochs, multi-sampling 5,
β = 0.8 (reconstruction objective) / β = 2 (latent objective). Isolation Forest:
100 estimators, contamination 5e-3, random_state 42.

### Three detection heads (paper Sec. 3.3)

| Head | Needs decoder? | Score | Threshold |
|------|----------------|-------|-----------|
| Reconstruction | yes | per-window MSE | max validation error × (1+γ) |
| Latent distance | no (encoder only) | Euclidean NN distance to training latent means (BallTree) | max validation distance × (1+γ) |
| Latent clustering | no (encoder only) | Isolation Forest anomaly score | IF decision boundary |

## Adaptation to this dataset

The paper's raw input is CAN arbitration ID + 8 payload bytes plus engineered
temporal features, with the binary arbitration ID mapped to a continuous vector by
an embedding layer (Sec. 3.1.1). **This repository's data is already *translated*
CAN** — decoded continuous signals:

```
speed_kmh, battery_level, throttle, brake, steering, gear, location_x, location_y
```

Because those signals are already continuous, the binary→continuous embedding step
and the payload-byte/entropy temporal features are unnecessary and are omitted. The
decoded signals are fed directly into the paper's message window (50 messages,
stride 20) and z-score standardised before the model, exactly as in Sec. 3.1.3.

### Activation note

The paper states the LSTMs use **ReLU**. A ReLU LSTM has an unbounded cell state
whose forward activations explode over length-50 sequences (reconstruction errors
of 1e11+), making it unusable here. The code therefore defaults to the standard,
numerically stable **tanh** (used by virtually all LSTM autoencoders) plus gradient
clipping, and exposes `--activation relu` for faithfulness experiments.

## Files

| File | Purpose |
|------|---------|
| `vae_model.py` | Flax β-VAE (`LSTMVAE`) + the three detection heads + save/load |
| `train_vae.py` | Centralized training on parquet; calibrates thresholds; saves a bundle |
| `evaluate_vae.py` | CLI: injects synthetic attacks; reports PR AUC + macro F1 per head |
| `evaluate_vae.ipynb` | Notebook version of the evaluation with PR curves and plots (like `mvp_v1`) |

## Setup

Uses the project virtual environment (already has jax, flax, optax, scikit-learn,
pandas, pyarrow). If starting fresh:

```bash
pip install "jax[cpu]" flax optax scikit-learn pandas pyarrow joblib
```

## Usage

Run from this directory (`Python scripts/vae_v1/`).

### Train

```bash
# Quick sanity run on one file
python train_vae.py ../../Data/new_data_processed/vehicle_1_20260321_145906.parquet --epochs 10

# Full pooled dataset, reconstruction objective (beta=0.8)
python train_vae.py ../../Data/new_data_processed --epochs 100

# Latent-space objective (beta=2.0) for the encoder-only heads
python train_vae.py ../../Data/new_data_processed --objective latent --epochs 100

# Faithful ReLU variant (may be unstable)
python train_vae.py ../../Data/new_data_processed --activation relu --epochs 100
```

The bundle is written to `checkpoints/` (weights `*_params.msgpack`, config/thresholds
`*_meta.json`, scaler `*_scaler.npz`, latent means `*_train_means.npy`, Isolation
Forest `*_iforest.joblib`).

### Evaluate

CLI:

```bash
python evaluate_vae.py ../../Data/new_data_processed --bundle checkpoints --name vae
```

Or open `evaluate_vae.ipynb` (run Jupyter from this directory so the local modules
import). Both report, for each of the three heads, the metrics the paper selected for
imbalanced detection (Sec. 4.1.1 / 4.3): **PR AUC** and **macro F1**, plus AUROC and
recall/FPR at the calibrated threshold, and a per-attack PR AUC breakdown. The
notebook adds overlaid PR curves, per-head score distributions, and a grouped
per-attack PR-AUC bar chart.

## Deployment note

For latent-space detection only the **encoder** is needed at inference (the paper's
main efficiency claim). The decoder weights are only required for the
reconstruction head. Whatever normalises incoming CAN data on the edge device must
use the saved `*_scaler.npz` mean/std, otherwise the calibrated thresholds are
invalid.
