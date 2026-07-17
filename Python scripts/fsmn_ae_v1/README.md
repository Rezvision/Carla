# fsmn_ae_v1 — FSMN Autoencoder IDS (translated CAN)

Implementation of the lightweight unsupervised model from:

> Y. Zhou, J. Zhang, G. Yang, *A Lightweight Unsupervised Intrusion Detection Model
> for In-Vehicle Edge Computing Based on FlexRay*, IEEE Access, 2026.

This is a sibling of `mvp_v1` (GRU autoencoder) and `vae_v1` (LSTM β-VAE). It trains
centrally on the translated (decoded) CAN telemetry in `Data/`.

## Model (paper Sec. III-A / III-B)

```
Input      :  (B, T, F) sequence window
Encoder    :  time-distributed Dense 64 → 32 → 16  (ReLU)
Memory     :  FSMN block  h_hat_t = Σ a_i · h_{t-i}   (Eq. 1, N=10)
              fused_t = ReLU(W h_t + Ŵ h_hat_t)      (Eq. 2)
Decoder    :  Dense 32 → 64 → F  (tanh hidden, linear output)
Loss       :  MSE reconstruction + λ · L1(code)     (Eq. 3 + sparsity)
Detection  :  per-window reconstruction MSE vs θ = μ + k·σ  (Eq. 4, k=2.5)
```

Unlike the VAE, FSMN-AE has a **single** detection head (reconstruction error only).
The FSMN memory block replaces a recurrent core, giving O(T·D·N) complexity instead of
O(T·D²) and far fewer parameters than GRU-/LSTM-based autoencoders.

## Adaptation to this dataset

The paper targets raw FlexRay frames (ID / payload / CRC) with a memory window matched
to the FlexRay static-segment cycle. **This repository's data is already translated CAN**
— decoded continuous signals:

```
speed_kmh, battery_level, throttle, brake, steering, gear, location_x, location_y
```

Changes from the paper for CAN:

1. **No FlexRay framing** — static/dynamic segment logic, CRC, and arbitration IDs are
   dropped; decoded signals are z-score standardised and fed directly to the encoder.
2. **Sequence windowing** matches `vae_v1` / `mvp_v1` (`WINDOW_SIZE=20`, `STRIDE=20`) so
   results are directly comparable across models.
3. **FSMN memory order N=10** is kept as the paper default but is a free hyperparameter
   (`--fsmn-order`); it no longer maps to FlexRay static cycles.

Everything else — unsupervised reconstruction, L1-sparse latent, dynamic threshold
θ = μ + k·σ — follows the paper.

## Files

| File | Purpose |
|------|---------|
| `fsmn_model.py` | Flax FSMN-AE + reconstruction detector + save/load |
| `train_fsmn.py` | Centralized training on parquet; calibrates θ; saves bundle |
| `evaluate_fsmn.py` | CLI: synthetic attacks; PR AUC + macro F1 |
| `evaluate_fsmn.ipynb` | Notebook evaluation with PR curves, score histograms, per-attack breakdown |

## Setup

Uses the project virtual environment (jax, flax, optax, scikit-learn, pandas, pyarrow).
If starting fresh:

```bash
pip install "jax[cpu]" flax optax scikit-learn pandas pyarrow matplotlib
```

## Usage

Run from this directory (`Python scripts/fsmn_ae_v1/`).

### Train

```bash
# Quick sanity run on one file
python train_fsmn.py ../../Data/new_data_processed/vehicle_1_20260321_145906.parquet --epochs 10

# Full pooled dataset (paper defaults: N=10, λ=0.001, k=2.5)
python train_fsmn.py ../../Data/new_data_processed --epochs 100

# Tune memory order for CAN periodicity
python train_fsmn.py ../../Data/new_data_processed --fsmn-order 15 --epochs 100
```

The bundle is written to `checkpoints/` (`*_params.msgpack`, `*_meta.json`,
`*_scaler.npz`).

### Evaluate

CLI:

```bash
python evaluate_fsmn.py ../../Data/new_data_processed --bundle checkpoints --name fsmn_ae
```

Or open `evaluate_fsmn.ipynb` (run Jupyter from this directory). Reports PR AUC, macro F1,
AUROC, recall/FPR at θ, per-attack PR AUC, parameter count, and streaming latency (p50/p99).

## Deployment note

Only the encoder + FSMN memory + decoder are needed at inference. Normalise incoming CAN
data with the saved `*_scaler.npz` mean/std; otherwise the calibrated θ is invalid.

On a typical run this model has ~7k parameters (~28 KiB float32) — compare against
`mvp_v1` (GRU-AE) and `vae_v1` (LSTM-VAE) using the footprint section in the notebook.
