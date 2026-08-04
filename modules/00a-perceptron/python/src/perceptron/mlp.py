"""A two-layer network small enough to read: 2 → h → 1, tanh, full-batch GD.

This is the smallest object that breaks the perceptron's ceiling. One hidden
layer bends the input space until XOR becomes linearly separable, and the
output unit — still a plain linear threshold — finishes the job. Module 02
scales exactly this code up; module 01 explains where the gradients come from.

Loss is mean squared error against ``y ∈ {−1, +1}``, which keeps the backward
pass to four lines and makes the analogy with least squares explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TinyMLP:
    W1: np.ndarray   # (2, h)
    b1: np.ndarray   # (h,)
    W2: np.ndarray   # (h, 1)
    b2: np.ndarray   # (1,)

    @staticmethod
    def init(n_in: int = 2, hidden: int = 2, seed: int = 7) -> "TinyMLP":
        rng = np.random.default_rng(seed)
        return TinyMLP(
            W1=rng.normal(0, 1.0, size=(n_in, hidden)),
            b1=np.zeros(hidden),
            W2=rng.normal(0, 1.0, size=(hidden, 1)),
            b2=np.zeros(1),
        )

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(hidden activations, output)``."""
        H = np.tanh(X @ self.W1 + self.b1)
        out = (H @ self.W2 + self.b2).ravel()
        return H, out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.forward(X)[1] > 0, 1, -1)

    def n_params(self) -> int:
        return sum(p.size for p in (self.W1, self.b1, self.W2, self.b2))


def train(
    model: TinyMLP, X: np.ndarray, y: np.ndarray, lr: float = 0.5,
    steps: int = 4000,
) -> tuple[TinyMLP, list[float], list[float]]:
    """Full-batch gradient descent on MSE. Returns ``(model, losses, accs)``."""
    n = len(X)
    losses: list[float] = []
    accs: list[float] = []
    for _ in range(steps):
        H = np.tanh(X @ model.W1 + model.b1)
        out = (H @ model.W2 + model.b2).ravel()
        err = out - y                                  # dL/dout, up to 2/n
        losses.append(float(err @ err / n))
        accs.append(float((np.sign(out) == y).mean()))

        gout = (2.0 / n) * err[:, None]                # (n, 1)
        gW2 = H.T @ gout
        gb2 = gout.sum(axis=0)
        gH = gout @ model.W2.T * (1.0 - H * H)         # tanh' = 1 − tanh²
        gW1 = X.T @ gH
        gb1 = gH.sum(axis=0)

        model.W1 -= lr * gW1
        model.b1 -= lr * gb1
        model.W2 -= lr * gW2
        model.b2 -= lr * gb2
    return model, losses, accs
