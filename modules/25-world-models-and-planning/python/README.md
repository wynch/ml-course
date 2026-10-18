# mlcourse-25-world-models-planning (Python lane)

Python code for module 25, world models and planning. See the module
overview one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing, 4 CPU threads), results I/O
- `src/grid.py`: the 8 × 8 grid world, its BFS, random-walk data and the test arenas
- `src/models.py`: the one-hot model, the JEPA, their training, the probe and the collapse meters
- `src/world.py`: section 1, arena statistics and the training data
- `src/train.py`: section 2, trains both models and reads what each learned; saves `../artifacts/models.pt`
- `src/plan.py`: section 3, random-shooting planning against BFS, random actions, a random-weight JEPA and a perfect model
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json` and the weights
- `tests/`: the solutions against `src/`; the results against the budget, each other and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
