# mlcourse-23-compression (Python lane)

Python code for module 23, compression. See the module overview one level up:
[`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), seeds, loaders, perplexity, results I/O
- `src/tiny_gpt.py`: loads module 04's trained char-GPT and builds fresh students of the same class
- `src/corpus.py`: reads the first two row groups of wikitext-103 by byte range
- `src/compress.py`: the rules by hand: magnitude and random masks, int8 symmetric, affine 4-bit (MLX's rule)
- `src/distil.py`: section 1, a half-size student from scratch against one distilled from the teacher
- `src/prune.py`: section 2, magnitude pruning of SmolLM2-360M-Instruct at 10, 30 and 50%, with a random control
- `src/quant.py`: section 3, int8 and 4-bit quantisation by hand: perplexity, bytes, speed
- `src/mlx_way.py`: section 4, `mlx_lm.convert` to 4 bits and the same measures in MLX
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts and MLX; the results against the budget and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
