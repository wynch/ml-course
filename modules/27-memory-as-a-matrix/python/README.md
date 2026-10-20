# mlcourse-27-memory-as-a-matrix (Python lane)

Python code for module 27, memory as a matrix. See the module overview one
level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: paths, the guard (time, peak RSS, peak MPS memory), results I/O
- `src/memory.py`: the four write rules (linear, decay, delta, gated delta), as a loop and in chunks
- `src/m18_selective.py`: module 18's selective recurrence, copied, to check against the decay rule
- `src/capacity.py`: section 1, writing pairs into one state with no training -> `../results/capacity.json`
- `src/forms.py`: section 2, loop against chunks, and their speed -> `../results/forms.json`
- `src/models.py`: the MQAR task and the small models (one shell, five mixers)
- `src/mqar.py`: sections 3 and 4, every training run in chunks of under five minutes -> `../results/mqar/*.json`, `../results/mqar.json`
- `src/library.py`: section 5, Qwen3.5-0.8B against Qwen3-0.6B on MLX -> `../results/library/*.json`, `../results/library.json`
- `src/figures.py`: redraws every figure from `../results/`
- `src/export_explorable.py`: writes the explorable's data block from `../results/`
- `tests/`: the solutions against `src/`, the two forms against each other, the results against the budget and the explorable

`.runs/` holds checkpoints of MQAR runs paused between chunks; it is not committed.

## Quickstart

```bash
uv sync
uv run pytest
```
