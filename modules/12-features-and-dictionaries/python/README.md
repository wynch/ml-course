# mlcourse-12-features-dictionaries (Python lane)

Python code for module 12, features and dictionaries. See the module overview
one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), loaders, results I/O
- `src/tiny_gpt.py`: loads module 04's trained char-GPT from its committed weight file
- `src/toy.py`: section 1, the toy model of superposition across ten sparsities
- `src/sae.py`: section 2, a TopK sparse autoencoder on the char-GPT's block-1 MLP neurons
- `src/corpus.py`: reads the first two row groups of wikitext-103 by byte range
- `src/gpt2_sae.py`: section 3, SAELens and TransformerLens on GPT-2 small, layer 8
- `src/steer.py`: section 4, steering against prompting on SmolLM2-360M-Instruct
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts; the results against the budget and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
