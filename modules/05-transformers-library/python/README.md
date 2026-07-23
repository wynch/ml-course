# mlcourse-05-transformers (Python lane)

Python code for Module 05 — The transformers library. See the module overview
one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py` — model id, device selection (MPS/CPU), cached loaders
- `src/pipelines.py` — first contact: `pipeline()` for text-gen, sentiment, zero-shot
- `src/anatomy.py` — walk the module tree, count parameters, draw the budget +
  tiny-GPT/SmolLM2/SmolLM3 comparison table
- `src/sampling_lab.py` — next-token distribution, temperature sweep, top-k vs top-p
- `src/logit_lens.py` — project each layer's hidden state through the output head;
  reusable `layerwise_predictions` / `decision_layer`
- `src/chat_template.py` — dissect the applied chat template to token boundaries
- `src/embeddings.py` — PCA projection of token embeddings + nearest neighbours
- `src/figures.py` — regenerate every figure embedded in the module README
- `app.py` — Gradio chat playground with a live next-token-distribution panel
- `tests/` — solution checks against the real model, and the app launch/close test

## Quickstart

```bash
uv sync
uv run pytest
```

The first run downloads SmolLM2-360M-Instruct (~700 MB) to the Hugging Face
cache (`~/.cache/huggingface`), not into this repo.
