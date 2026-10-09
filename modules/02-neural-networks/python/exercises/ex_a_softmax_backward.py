"""Exercise (a) — derive and implement the softmax + cross-entropy backward.

The forward pass is given. Your job is the backward pass: fill in
``SoftmaxCrossEntropy.backward`` so that the numerical gradient check at the
bottom passes.

Run:
    uv run python exercises/ex_a_softmax_backward.py

You should see "GRADCHECK: FAIL" until you implement it correctly, then PASS.

------------------------------------------------------------------------------
Derivation to do on paper first (then translate to two lines of code):

  Let z be the logits (N, C), p = softmax(z), and y the integer labels.
  Loss L = -(1/N) * sum_i log p[i, y_i].

  Softmax:   p_k = exp(z_k) / sum_j exp(z_j)
  Its Jacobian:  d p_k / d z_m = p_k (delta_km - p_m)

  Cross-entropy for one row: L_i = -log p_{y_i}.
  Chain-rule through the softmax and the sum collapses to a famously clean
  result:
                    d L_i / d z_m = p_m - [m == y_i]

  Averaged over the batch:
                    d L / d z = (p - onehot(y)) / N

  That's the whole gradient. Implement it below.
------------------------------------------------------------------------------
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
        """Return dL/d(logits), shape (N, C).

        # TODO(you): implement the fused softmax+cross-entropy gradient.
        # Hint: start from a copy of self._probs, subtract 1 from the entry of
        # the true class in each row, then divide by the batch size N.
        """
        raise NotImplementedError("implement the softmax+cross-entropy backward")


def gradient_check():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6, 4))
    y = rng.integers(0, 4, size=6)

    loss = SoftmaxCrossEntropy()
    loss.forward(logits, y)
    try:
        analytic = loss.backward()
    except NotImplementedError as e:
        print(f"backward not implemented yet: {e}")
        print("GRADCHECK: FAIL")
        return

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
