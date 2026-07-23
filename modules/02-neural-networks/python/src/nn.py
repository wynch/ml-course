"""A tiny neural-network library in pure NumPy.

Everything here is written from scratch so you can see exactly where each
gradient comes from. No autograd engine, no PyTorch: every layer implements a
``forward`` and a ``backward`` by hand, and the backward passes are derived in
the README (and again, briefly, in the docstrings below).

Design
------
* A ``Layer`` has ``forward(x)`` and ``backward(grad_out)``.
* ``forward`` caches whatever the backward pass needs on ``self``.
* ``backward`` receives dL/d(output) and returns dL/d(input), while stashing
  parameter gradients on the layer for the optimizer to consume.
* An ``MLP`` chains layers; the loss (softmax + cross-entropy) is fused so the
  gradient at the logits is the famously clean ``(softmax - onehot) / N``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class Linear:
    r"""Fully-connected layer: ``y = x @ W + b``.

    Shapes (N = batch size):
        x : (N, in_features)
        W : (in_features, out_features)
        b : (out_features,)
        y : (N, out_features)

    Backward (chain rule).  Given ``g = dL/dy`` with shape (N, out):
        dL/dW = x^T @ g          (in, out)
        dL/db = sum over batch g (out,)
        dL/dx = g @ W^T          (N, in)
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        # He (Kaiming) initialization: variance 2/fan_in keeps ReLU
        # activations from vanishing or exploding across layers.
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, scale, size=(in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db


class ReLU:
    r"""Rectified linear unit: ``y = max(0, x)``.

    Backward: the gradient passes through where the input was positive and is
    zeroed elsewhere, i.e. ``dL/dx = dL/dy * [x > 0]``.
    """

    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask

    def params_and_grads(self):
        return iter(())  # no parameters


# ---------------------------------------------------------------------------
# Softmax + cross-entropy (fused)
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the standard max-subtraction for stability."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true classes.

    ``probs`` is (N, C) softmax output, ``y`` is (N,) integer labels.
    """
    n = y.shape[0]
    # clip to avoid log(0); the model can be very confident on easy examples
    p = np.clip(probs[np.arange(n), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


class SoftmaxCrossEntropy:
    r"""Fused softmax + cross-entropy loss.

    Why fuse?  If ``p = softmax(z)`` and the loss is ``-log p[y]``, the algebra
    collapses to a beautifully simple gradient at the logits:

        dL/dz = (p - onehot(y)) / N

    Deriving it separately and multiplying the two Jacobians gives the same
    thing but is numerically noisier and much slower.  This is exercise (a).
    """

    def __init__(self):
        self._probs = None
        self._y = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        self._probs = softmax(logits)
        self._y = y
        return cross_entropy(self._probs, y)

    def backward(self) -> np.ndarray:
        n = self._y.shape[0]
        grad = self._probs.copy()
        grad[np.arange(n), self._y] -= 1.0
        return grad / n


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


class MLP:
    """A multi-layer perceptron: a list of layers applied in sequence.

    Build it from a list of layer sizes, e.g. ``MLP([784, 256, 10])`` is
    Linear(784, 256) -> ReLU -> Linear(256, 10).  Any number of hidden layers
    works (that is exercise (b)).
    """

    def __init__(self, sizes, rng: np.random.Generator):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng))
            if i < len(sizes) - 2:  # ReLU between hidden layers, not after logits
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params_and_grads(self):
        for layer in self.layers:
            yield from layer.params_and_grads()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of correctly classified rows."""
    return float((model.predict(x) == y).mean())
