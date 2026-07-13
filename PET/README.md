# PET_Code

GRU-based intrusion detection for vehicle CAN bus telemetry, with differential
privacy (input / output / gradient perturbation) as a defence. Studies the
privacy–utility tradeoff.

## Layout
- `notebooks/` — current model code (`workflow.ipynb`) + saved `models/model.pth`
- `data/CARLA_processed/` — working dataset, 103 sessions / 3.16M rows (gitignored)
- `src/` — refactored, importable code
- `tests/` — pytest tests

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch opacus pytest pandas pyarrow scikit-learn matplotlib
```

## Goal (this week)
Baseline GRU autoencoder IDS → input, output, and gradient (DP-SGD/Opacus)
perturbation variants over several noise levels → one privacy-vs-utility comparison.

See `CLAUDE.md` for full project context.
