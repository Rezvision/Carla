#!/usr/bin/env python3
"""
train_central.py — centralized GRU training on CARLA data.

Delegates to shared ``ids_core`` (dataset=carla, model=gru). The live edge
client in ``fed_client_jax.py`` is unchanged.

Run from Python scripts/mvp_v1/:

    python train_central.py ../../Data/carla/processed/new --epochs 20 --stride 20

Writes to ``/tmp/fed_ids_checkpoints`` by default so ``fed_client_jax`` can load
``central`` without path changes. For the shared research tree use:

    python -m experiments.train --dataset carla --model gru --name central
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ids_core.train import main as _core_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve positional data arg; inject dataset/model if missing.
    if "--dataset" not in argv:
        argv = ["--dataset", "carla", *argv]
    if "--model" not in argv:
        # Insert after --dataset carla
        out = []
        i = 0
        while i < len(argv):
            out.append(argv[i])
            if argv[i] == "--dataset" and i + 1 < len(argv):
                out.append(argv[i + 1])
                out.extend(["--model", "gru"])
                i += 2
                continue
            i += 1
        argv = out
    # Historical default checkpoint name + live-client outdir
    if "--name" not in argv:
        argv.extend(["--name", "central"])
    if "--outdir" not in argv:
        argv.extend(["--outdir", "/tmp/fed_ids_checkpoints"])
    return _core_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
