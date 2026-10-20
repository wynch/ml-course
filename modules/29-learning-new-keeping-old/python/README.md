# mlcourse-29-learning-new-keeping-old (Python lane)

Python code for module 29, learning something new without losing the old.
See the module overview one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing, 4 CPU threads), pinned datasets, results I/O
- `src/gpt.py`: module 04's char-level GPT, copied unchanged
- `src/data.py`: the two texts (Shakespeare, GSM8K's solutions), their shared vocabulary, batches, the fixed held-out windows
- `src/lora.py`: LoRA by hand, a frozen linear layer plus a low-rank change
- `src/forget.py`: sections 1 to 3; trains the start model (`../artifacts/start.pt`), then the six arms, 3 seeds each, in chunks
- `src/plasticity.py`: section 4; 800 shuffled-pixel FashionMNIST tasks, backprop against continual backprop, in chunks
- `src/library.py`: section 5; PEFT LoRA on SmolLM2-360M-Instruct, the 500-item probe, the memory of one step
- `src/analyse.py`: reads every run, writes `../results/forget.json`, `plasticity.json` and `library.json`, scores each written expectation
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/`
- `tests/`: data, LoRA, measures and solutions against `src/`; the results against the budget, each other and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
