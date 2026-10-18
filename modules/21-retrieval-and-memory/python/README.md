# mlcourse-21-retrieval-and-memory (Python lane)

Python code for module 21, retrieval and memory. See the module overview one
level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), model loaders, results I/O
- `src/data.py`: SQuAD v1.1 validation, deduplicated paragraphs, and the 200-question sample
- `src/bm25.py`: the tokeniser, an inverted index and BM25; a no-index scan for checks
- `src/squad_metric.py`: SQuAD's normalisation, exact match and F1
- `src/embed.py`: section 1, MiniLM and SmolLM2 mean-pool vectors for every paragraph and question
- `src/retrieval.py`: section 2, recall@k for BM25 and both cosines
- `src/rag.py`: section 3, SmolLM2-360M-Instruct answers 200 questions; replies cached, then scored
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts and hand-worked values; the results against the budget and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
