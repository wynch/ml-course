"""Gradient check for the whole library: Linear, ReLU and softmax + cross-entropy.

Builds a small MLP, runs the hand-written backward pass, and compares every
parameter gradient (and the input gradient) with central finite differences.
Run:

    uv run python gradcheck.py
Expected: GRADCHECK: PASS
"""

from __future__ import annotations

import numpy as np

from src.nn import MLP, SoftmaxCrossEntropy


def main(seed: int = 0, eps: float = 1e-6, tol: float = 1e-6) -> float:
    rng = np.random.default_rng(seed)
    model = MLP([5, 7, 6, 3], rng)  # Linear -> ReLU -> Linear -> ReLU -> Linear
    x = rng.normal(size=(8, 5))
    y = rng.integers(0, 3, size=8)
    loss_fn = SoftmaxCrossEntropy()

    def loss_at() -> float:
        return loss_fn.forward(model.forward(x), y)

    loss_at()
    grad_in = loss_fn.backward()
    for layer in reversed(model.layers):
        grad_in = layer.backward(grad_in)
    analytic = [g.copy() for _, g in model.params_and_grads()] + [grad_in]

    tensors = [p for p, _ in model.params_and_grads()] + [x]
    worst = 0.0
    for t, a in zip(tensors, analytic):
        num = np.zeros_like(t)
        for i in np.ndindex(t.shape):
            orig = t[i]
            t[i] = orig + eps
            lp = loss_at()
            t[i] = orig - eps
            lm = loss_at()
            t[i] = orig
            num[i] = (lp - lm) / (2 * eps)
        rel = np.abs(a - num).max() / max(np.abs(a).max() + np.abs(num).max(), 1e-12)
        worst = max(worst, rel)

    print(f"max relative error over {len(tensors)} tensors: {worst:.2e}")
    print("GRADCHECK:", "PASS" if worst < tol else "FAIL")
    return worst


if __name__ == "__main__":
    main()
