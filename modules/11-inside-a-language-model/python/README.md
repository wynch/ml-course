# mlcourse-11-inside-lm (Python lane)

Python code for module 11, inside a language model. See the module overview
one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), model loaders, results I/O
- `src/tiny_gpt.py`: loads module 04's trained char-GPT from its committed weight file
- `src/hooks.py`: section 1, a reading hook and an editing hook on one neuron
- `src/probe.py`: section 2, a logistic-regression probe per layer on SST-2, with controls
- `src/patching.py`: section 3, the Eiffel/Colosseum patching sweep and the MPS-against-CPU check
- `src/corpus.py`: reads the first 1M tokens of wikitext-103 by Parquet row group
- `src/poles.py`: section 4, top contexts on both poles for three SmolLM2 neurons
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts; the results against the budget and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
