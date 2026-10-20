# mlcourse-33-text-by-denoising (Python lane)

Python code for module 33, text by denoising. See the module overview one
level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks at 0.3, peak memory, timing, 4 CPU threads, chunk length), results I/O
- `src/data.py`: tiny-shakespeare at the character level, the fixed held-out blocks, the word check
- `src/model.py`: module 04's char-GPT with a causal switch, a [MASK] row, attention masks and a KV cache
- `src/diffusion.py`: the masking process, the three losses, the scores (exact, ELBO, fixed-order), the samplers
- `src/flow2d.py`: section 1, flow matching against DDPM on module 09's spiral
- `src/train.py`: every model, in chunks under 5 minutes; `--export` writes `../artifacts/models.pt`
- `src/evaluate.py`: section 2, held-out likelihoods and the infill test
- `src/sample.py`: section 3, 1 to 128 steps, two rules; the traces for the explorable
- `src/block.py`: section 4, block diffusion with and without the KV cache
- `src/repeat.py`: section 5, the data-repeat summary from the run files
- `src/library.py`: section 6, Meta's `flow_matching` and LLaDA's sampler against our code
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against `src/`; the results against the budget, each other and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
