# mlcourse-03-tokenization (Python lane)

Python code for Module 03 — Tokenization. See the module overview one level up:
[`../README.md`](../README.md).

## Layout

- `src/bpe.py` — byte-level BPE from scratch (`train` / `encode` / `decode`)
- `src/hf_way.py` — train a BPE with the `tokenizers` library and compare against
  SmolLM3's production tokenizer
- `src/figures.py` — generate every figure embedded in the module README
- `app.py` — Gradio tokenization playground
- `tests/` — round-trip properties, the Zig cross-language check, the app
  launch/close test, and the exercise-solution checks

## Quickstart

```bash
uv sync
uv run pytest
```
