# experiments — train, compare, evaluate, artifacts

```bash
cd "Python scripts"

# Train
python -m experiments.train --dataset carla  --model gru  --epochs 20
python -m experiments.train --dataset kaggle --model tet  --epochs 50

# Compare
python -m experiments.compare --dataset carla  ../Data/carla/processed/new
python -m experiments.compare --dataset kaggle ../Data/kaggle/processed

# Evaluate (Kaggle failure labels)
python experiments/evaluate_kaggle.py ../Data/kaggle/processed --model gru
```

## Artifacts

```
experiments/
  checkpoints/{carla,kaggle}/
  figures/{carla,kaggle}/
  notebooks/
  train.py
  compare.py
  compare_carla.py
  compare_kaggle.py
  evaluate_kaggle.py
```
