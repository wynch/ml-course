# mlcourse-14-how-training-finds-a-solution (Python lane)

Python code for module 14, how training finds a solution. See the module
overview one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), FashionMNIST loader, module 02's MLP in torch, results I/O
- `src/optim_toy.py`: section 1, SGD, momentum and Adam from scratch on two 2-D losses, 37 learning rates each
- `src/optim_mlp.py`: section 2, the three on the MLP, four learning rates, constant against warmup + cosine
- `src/hf_way.py`: section 2, `get_cosine_schedule_with_warmup` checked against ours; `Trainer` with `TrainingArguments(optim=...)`
- `src/landscape.py`: section 3, small against large batch: 1-D and 2-D loss slices, top Hessian eigenvalue
- `src/grok.py`: section 4, grokking on addition mod 97, weight decay 1.0 or 0
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts and transformers; the results against the budget, the grokking rule and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
