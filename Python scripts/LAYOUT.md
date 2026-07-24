# Project layout (multi-model IDS)

```
Carla/
├── Data/
│   ├── carla/processed/
│   │   ├── new/                  # main CARLA corpus (default)
│   │   └── base/                 # smaller / archived CARLA set
│   └── kaggle/
│       ├── raw/                  # CSVs
│       └── processed/            # per-vehicle parquets
│
└── Python scripts/
    ├── models/                   # ARCHITECTURE ONLY
    │   ├── gru/  vae/  fsmn/  tet/
    ├── ids_core/                 # shared profiles, data, adapters, trainer
    ├── datasets/                 # preprocess / augment only
    │   └── kaggle/
    ├── experiments/              # CLIs + checkpoints + figures + notebooks
    ├── mvp_v1/                   # live client (fed_client_jax) + thin train shim
    ├── edge/                     # Pi / live deployment
    ├── server/                   # federation / dashboards
    └── sim/                      # (new) simulation utilities
```

**Rules**

- New model → `models/<name>/` + adapter in `ids_core/models/`
- New dataset → `ids_core/profiles/<name>.py` + optional `datasets/<name>/`
- Do not create a new per-model package that copies train/eval
- Do not change `mvp_v1/fed_client_jax.py` feature contract without a deploy plan
