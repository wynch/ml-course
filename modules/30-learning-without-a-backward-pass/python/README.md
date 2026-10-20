# mlcourse-30-learning-without-a-backward-pass (Python lane)

Python code for module 30, learning without a backward pass. See the module
overview one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing, 4 CPU threads), FashionMNIST, results I/O
- `src/rules.py`: the six learners (backprop, checkpointing, MeZO, forward-forward, Mono-Forward, predictive coding), Adam and SGD by hand
- `src/memory.py`: section 1, what each rule holds while it trains, each in a fresh process, with a live-tensor meter
- `src/train.py`: section 2, every rule on FashionMNIST, 3 seeds, in chunks of at most 4.5 minutes
- `src/credit.py`: section 3, how predictive coding's error reaches the lower layers
- `src/library.py`: section 4, MeZO against LoRA backprop on SmolLM2-360M-Instruct for SST-2, in chunks
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the rules against autograd; the solutions against `src/`; the results against the budget, each other and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
