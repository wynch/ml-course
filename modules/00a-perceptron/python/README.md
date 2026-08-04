# mlcourse-00a-perceptron (Python lane)

The from-scratch perceptron and least-squares code for Module 00a. See the
module [README](../README.md) for the full walkthrough.

Layout:

- `src/perceptron/mp_neuron.py` — McCulloch–Pitts threshold logic (no learning)
- `src/perceptron/rosenblatt.py` — the learning rule, the mistake trace, and
  `R`, `γ`, `(R/γ)²` via Frank–Wolfe
- `src/perceptron/lstsq.py` — normal equations, gradient descent, the projection
  geometry, the stability edge
- `src/perceptron/mlp.py` — a 2→2→1 tanh network, forward and backward by hand
- `src/perceptron/data.py` — every seeded dataset in the module
- `scripts/` — figure-producing scripts, run with `uv run scripts/<name>.py`
- `tests/` — cross-checks against numpy's solvers and scikit-learn

Each script also writes a `run_*.json` blob beside this file; those are the
numbers the Zig lane, the tests and the explorable check themselves against.

Quick start:

```bash
uv run scripts/train_perceptron.py
uv run pytest -q
```
