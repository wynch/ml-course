# mlcourse-02-neural-networks

Pure-NumPy neural-network library and training scripts for module 02. See the
[module README](../README.md) for the full walkthrough, theory, and figures.

## Quick start

```bash
uv sync
uv run python train_toy.py       # decision-boundary GIF (2D toy data)
uv run python train_fashion.py   # FashionMNIST MLP -> >=85% test acc + figures
uv run python bench_numpy.py     # BLAS matmul timing (pairs with ../zig)
```

## Layout

- `src/` — the library: `nn.py` (layers, loss, MLP), `optim.py` (SGD, Adam),
  `train.py` (training loop), `data.py` (datasets), `plots.py` (figures).
- `exercises/` — skeletons with `# TODO(you):` markers.
- `solutions/` — verified reference answers.

All figures are written to `../figures/`.
