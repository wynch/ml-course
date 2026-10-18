# mlcourse-20-in-context-learning (Python lane)

Python code for module 20, in-context learning. See the module overview one
level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), loaders, results I/O
- `src/icl.py`: the regression task, the transformer, and four classic rules (least squares, ridge, 3-NN, one gradient step)
- `src/train_icl.py`: sections 1 and 2, the main net and the shuffled-y control net
- `src/gd_demo.py`: section 3, implied weights and layer probes (a demo)
- `src/fewshot.py`: section 4, few-shot SST-2 on SmolLM2-360M-Instruct
- `src/line_net.py`: the small net on lines that the explorable runs
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts; the prompt-permutation test; the results against the budget and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
