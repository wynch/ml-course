# mlcourse-22-reasoning-at-test-time (Python lane)

Python code for module 22, reasoning at test time. See the module overview
one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: paths, pinned model and dataset revisions, the guard (time, peak RSS, peak Metal memory), results I/O
- `src/gsm8k.py`: the problems, the 8-shot prompt, the answer reader, the vote
- `src/generate.py`: every model call (answer only, greedy, 8 samples, self-grades), cached to `../results/generations/`
- `src/analyse.py`: sections 1–3 from the cached generations alone -> `../results/analysis.json`
- `src/device_check.py`: one short forward pass on the GPU and the CPU
- `src/figures.py`: redraws every figure from `../results/analysis.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/`
- `tests/`: the answer reader's 50 known replies; the solutions against `src/`; the results against the caches, the budget and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
