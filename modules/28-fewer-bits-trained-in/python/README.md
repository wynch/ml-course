# mlcourse-28-fewer-bits-trained-in (Python lane)

Python code for module 28, fewer bits, trained in. See the module overview one level up:
[`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), seeds, SmolLM2 loaders, perplexity ids, results I/O
- `src/tiny_gpt.py`: module 04's char-GPT class, its trained weights, and fresh models of any width and depth
- `src/chars.py`: the first 20,000,000 characters of wikitext-103 as character ids, and the held-out halves
- `src/corpus.py`: the wikitext slice SmolLM2's perplexity reads (module 23's)
- `src/bitlinear.py`: BitLinear by hand: absmean ternary weights, 8-bit inputs, the straight-through gradient, byte counts
- `src/compress.py`: module 23's affine rule (MLX's), used to round the float models
- `src/train.py`: one training run, float or ternary, that can pause at a deadline and resume
- `src/matched.py`: section 1, four arms at two file sizes, three shapes and a tuned rate each, in chunks of at most 4.5 minutes
- `src/followup.py`: section 1's follow-up, written after its results: the one-block shape gets its own rate
- `src/export.py`: section 2, the bf16 cast at export and the fine-tune it undoes
- `src/kvquant.py`: the KV cache rounded by hand in MLX: per token or per channel, with or without a Hadamard rotation
- `src/kv_hand.py`: section 3, SmolLM2-360M with its cache at 8, 4, 3 and 2 bits; key outliers
- `src/kv_mlx.py`: section 4, mlx-lm's `QuantizedKVCache`: perplexity, bytes, speed
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts and MLX; the results against the budget, the protocol and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
