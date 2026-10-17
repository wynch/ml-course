# mlcourse-18-beyond-attention (Python lane)

Python code for module 18, beyond attention. See the module overview one level
up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), module 04's char data, results I/O
- `src/models.py`: the GRU, the two SSMs (fixed and selective recurrence) and module 04's GPT with a KV cache; each has a parallel forward and a one-token `step`
- `src/train.py`: the one training recipe every model uses, by steps or by seconds
- `src/gru.py`: section 1, a GRU at the tiny GPT's size
- `src/ssm.py`: section 2, a diagonal linear recurrence, then a selective one
- `src/compare.py`: section 3, GPT, GRU and SSM trained for the same wall time (`--budget`)
- `src/decode.py`: section 3, memory and time per generated token up to 4,096 tokens
- `src/hf_compare.py`: section 4, Mamba-130M against GPT-2 small through transformers
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the code; the results against the budget and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
