# mlcourse-01-autograd (Python lane)

The Python autograd engine for Module 01. See the module
[README](../README.md) for the full walkthrough and
[ALGORITHM.md](../ALGORITHM.md) for the Python/Zig comparison.

Layout:

- `src/micrograd/engine.py` — the `Value` scalar autograd engine
- `src/micrograd/nn.py` — `Neuron` / `Layer` / `MLP`
- `src/micrograd/draw.py` — computation-graph renderer (graphviz or matplotlib)
- `scripts/` — figure-producing / training scripts, run with `uv run scripts/<name>.py`

Quick start:

```bash
uv run scripts/train_moons.py
```
