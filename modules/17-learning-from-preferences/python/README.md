# mlcourse-17-learning-from-preferences (Python lane)

Python code for module 17, learning from preferences. See the module overview
one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), loaders, results I/O
- `src/data.py`: reads nine row groups of `trl-lib/ultrafeedback_binarized` by byte range, filters and splits the pairs
- `src/bt_synthetic.py`: section 1a, Bradley-Terry by hand on synthetic pairs
- `src/reward_head.py`: section 1b, a linear reward head on SmolLM2-360M-Instruct's layer-18 residuals; the MPS-vs-CPU check
- `src/dpo_by_hand.py`: section 2, the DPO loss by hand against TRL on one batch
- `src/dpo_train.py`: section 3, DPO with TRL + LoRA, `--config preview` or `--config full`
- `src/length_bias.py`: section 4, length bias in the data, the reward head and DPO
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts; the results against the budget and the explorable
- `outputs/` (git-ignored): the trained LoRA adapters

## Quickstart

```bash
uv sync
uv run pytest
```
