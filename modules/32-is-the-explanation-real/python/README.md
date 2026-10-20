# mlcourse-32-is-the-explanation-real (Python lane)

Python code for module 32, Is the explanation real? See the module overview
one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS cap at 0.3 of Metal's working set, peak memory, timing), paths, results I/O
- `src/data.py`: TinyStories (one pinned file), GPT-2's tokenizer cut to a 4,096-token vocabulary
- `src/model.py`: the two-layer transformer with TopK-sparse MLPs (the truth), the dense baseline, the random copy
- `src/train.py`: section 1, trains both models in chunks of under 5 minutes
- `src/dicts.py`: SAEs and transcoders on block 1's MLP, trained and random model, seeds 0 to 2, in chunks
- `src/scores.py`: best cosines (MMCS), seed matching, the detection score and token R²
- `src/analyse_dicts.py`: sections 2 to 4, scores every dictionary and unit
- `src/attribution.py`: section 5, the attribution graph and the ablations that test it
- `src/library.py`: section 6, SAELens on GPT-2 small and on a random-weight GPT-2
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against `src/`; the results against the budget, each other and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
