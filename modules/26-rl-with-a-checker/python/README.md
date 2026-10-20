# mlcourse-26-rl-with-a-checker (Python lane)

Python code for module 26, RL with a checker. See the module overview one
level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing, 4 CPU threads), results I/O
- `src/gpt.py`: module 04's char-level GPT, copied unchanged
- `src/task.py`: 3-digit sums, the two reply styles, the checker, the test split, the pass@k estimator
- `src/rollout.py`: batched sampling, rewards and per-character log-probabilities
- `src/sft.py`: section 1, trains the base model and saves `../artifacts/base.pt`
- `src/rl.py`: sections 2 and 3, GRPO, Dr. GRPO, RLOO and the random-reward controls, 3 seeds each, in chunks
- `src/analyse.py`: reads `../results/runs/` and writes `../results/rl.json`, with each written expectation scored
- `src/library.py`: section 4, TRL's `GRPOTrainer` with LoRA on SmolLM2-360M-Instruct, in chunks
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/`
- `tests/`: the task, checkers and solutions against `src/`; the results against the budget, each other and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
