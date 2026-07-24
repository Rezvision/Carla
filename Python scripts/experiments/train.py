#!/usr/bin/env python3
"""
Unified trainer entrypoint.

    cd "Python scripts"
    python -m experiments.train --dataset carla  --model gru     --epochs 20
    python -m experiments.train --dataset kaggle --model fsmn    --epochs 50
    python -m experiments.train --dataset can    --model can_gru --epochs 20

Also available as ``python -m ids_core.train ...``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ids_core.train import main

if __name__ == "__main__":
    raise SystemExit(main())
