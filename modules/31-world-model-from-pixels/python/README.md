# mlcourse-31-world-model-from-pixels (Python lane)

Python code for module 31, a world model from pixels. See the module
overview one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing, 4 CPU threads), results I/O
- `src/tworoom.py`: the two-room world (a port of stable-worldmodel's TwoRoom), its 64-pixel frames, the expert, the data and the test tasks
- `src/lewm.py`: the encoder, the predictor, SIGReg, the two-term loss, BatchNorm recalibration, probes and meters
- `src/world.py`: section 1, the world and its data in numbers
- `src/train.py`: section 2, four recipes x three seeds in chunks of under 5 minutes; read-outs; `artifacts/lewm_planner.pt`
- `src/plan.py`: section 3, CEM, random shooting and gradient planning against fixed agents, in chunks
- `src/grid.py`, `src/grid_models.py`, `src/grid_plan.py`: module 25's grid, models and planner, copied
- `src/grid_terms.py`: section 4, module 25's JEPA with PLDM's and LeWM's terms, one at a time
- `src/library.py`: section 5, stable-worldmodel's env, losses and CEM solver against ours
- `src/figures.py`: redraws every figure from `../results/`
- `src/export_explorable.py`: writes the explorable's data block from `../results/` and the planner artifact
- `tests/`: the solutions against `src/`; the results against the budget, each other, the expectations and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
