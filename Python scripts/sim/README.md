# sim — CARLA / ECU / attack tooling

Simulation and bus-injection scripts currently still live at the
`Python scripts/` root (e.g. `carla_client*.py`, `attack_simulation*.py`,
`vTCU_python*.py`) so existing deploy/docs paths keep working.

New simulation utilities should go here. Do not put model architecture or
training code in this folder — use `models/` and `ids_core/` instead.
