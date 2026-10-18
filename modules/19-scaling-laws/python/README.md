# mlcourse-19-scaling-laws (Python lane)

Python code for module 19, scaling laws. See the module overview one level up:
[`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), the shared recipe, results I/O
- `src/char_gpt.py`: module 04's char-GPT, copied so this module stands alone
- `src/corpus.py`: the first 5M characters of wikitext-103 and the character vocabulary
- `src/train.py`: section 1, one size per call; held-out loss at twelve points
- `src/fit.py`: section 2, L(N), L(D), a joint L(N, D), the fitter's check, unigram and bigram baselines
- `src/frontier.py`: section 3, the lowest loss at each compute budget
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts; the results against the budget and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
