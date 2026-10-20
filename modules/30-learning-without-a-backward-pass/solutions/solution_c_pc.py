"""Solution (c): one step of predictive coding's inference, by hand.

    uv run python ../solutions/solution_c_pc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import torch


def node_grads(layers, xs, y):
    """∂F/∂x_l for each hidden node l = 1 .. L−1, with
    F = Σ_hidden ½‖x_l − μ_l‖² + CE(μ_L, y), μ_l = φ(x_{l−1}) W_lᵀ + b_l,
    φ the identity on the input and ReLU elsewhere. Summed over the batch."""
    L = len(layers)
    phi = lambda l, x: x if l == 0 else torch.relu(x)
    mu = lambda l: phi(l - 1, xs[l - 1]) @ layers[l - 1][0].T + layers[l - 1][1]
    delta = [None] + [-(xs[l] - mu(l)) for l in range(1, L)]          # ∂F/∂μ_l on hidden layers
    p = torch.softmax(mu(L), dim=1)
    p[torch.arange(len(y)), y] -= 1.0
    delta.append(p)                                                   # ∂F/∂μ_L = softmax − onehot
    return [(xs[l] - mu(l)) + (xs[l] > 0) * (delta[l + 1] @ layers[l][0]) for l in range(1, L)]


def infer(layers, x, y, T: int, gamma: float = 0.1):
    """Start from the feedforward pass, then T steps of x_l ← x_l − γ ∂F/∂x_l."""
    xs = [x]
    for l in range(1, len(layers)):
        w, b = layers[l - 1]
        xs.append((xs[-1] if l == 1 else torch.relu(xs[-1])) @ w.T + b)
    for _ in range(T):
        g = node_grads(layers, xs, y)
        xs = [xs[0]] + [xl - gamma * gl for xl, gl in zip(xs[1:], g)]
    return xs


def main() -> None:
    import rules as R

    _, _, x_te, y_te = common.load_fashion()
    pc = R.PredictiveCoding((784,) + (256,) * 5 + (10,), 0)
    x, y = x_te[:256], y_te[:256]
    start = infer(pc.layers, x, y, 0)
    for T in range(1, 7):
        xs = infer(pc.layers, x, y, T)
        moved = [int((a - b).abs().max() > 0) for a, b in zip(xs[1:], start[1:])]
        print(f"after {T} steps, hidden layers 1-5 moved: {moved}")


if __name__ == "__main__":
    main()
