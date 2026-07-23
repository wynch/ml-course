"""Solution (a) — softmax + cross-entropy backward.

Run:
    uv run python solutions/ex_a_softmax_backward.py
Expected: GRADCHECK: PASS
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    def __init__(self):
        self._probs = None
        self._y = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        self._probs = softmax(logits)
        self._y = y
        n = y.shape[0]
        p = np.clip(self._probs[np.arange(n), y], 1e-12, 1.0)
        return float(-np.log(p).mean())

    def backward(self) -> np.ndarray:
        # dL/dz = (p - onehot(y)) / N
        n = self._y.shape[0]
        grad = self._probs.copy()
        grad[np.arange(n), self._y] -= 1.0
        return grad / n


def gradient_check():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6, 4))
    y = rng.integers(0, 4, size=6)

    loss = SoftmaxCrossEntropy()
    loss.forward(logits, y)
    analytic = loss.backward()

    eps = 1e-6
    numeric = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            orig = logits[i, j]
            logits[i, j] = orig + eps
            lp = loss.forward(logits, y)
            logits[i, j] = orig - eps
            lm = loss.forward(logits, y)
            logits[i, j] = orig
            numeric[i, j] = (lp - lm) / (2 * eps)

    rel = np.abs(analytic - numeric).max() / (
        np.abs(analytic).max() + np.abs(numeric).max() + 1e-12
    )
    print(f"max relative error: {rel:.2e}")
    print("GRADCHECK:", "PASS" if rel < 1e-5 else "FAIL")


if __name__ == "__main__":
    gradient_check()
