# mlcourse-04-attention

PyTorch (MPS) tiny char-level GPT and the numpy attention warm-up for module 04.
See the [module README](../README.md) for the full walkthrough, theory, and figures.

## Quick start

```bash
uv sync
uv run python scripts/step1_numpy_attention.py   # numpy SDPA + causal-mask figure
uv run python scripts/train.py                    # train the tiny GPT on MPS (~4-8 min on M5)
uv run python scripts/figures.py                  # all figures + attention gif from the trained model
uv run python scripts/export_weights.py           # write artifacts/ for the module-07 Zig capstone
```

## Layout

- `src/model.py` — `GPT`, `Block`, `MultiHeadAttention`, `MLP` from nn.Module primitives (no `nn.Transformer`).
- `src/data.py` — char tokenizer + batching for tiny-shakespeare.
- `src/config.py` — shared paths, device selection, hyper-parameters.
- `scripts/` — step 1 (numpy), training, figure generation, weight export.
- `../exercises/` — skeletons with `# TODO(you):` markers.
- `../solutions/` — verified reference answers.

Figures are written to `../figures/`, checkpoints to `../models/` (git-ignored),
and the exported model to `../artifacts/` (committed).
