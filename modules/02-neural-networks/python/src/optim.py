"""Optimizers: mini-batch SGD and Adam, both from scratch.

An optimizer owns the update rule. It is handed the list of (param, grad)
pairs each step and mutates the params in place. The model produced the grads;
the optimizer decides how far and in what direction to step.
"""

from __future__ import annotations

import numpy as np


class SGD:
    """Vanilla stochastic gradient descent: ``p <- p - lr * grad``.

    Optionally with classical momentum, which accumulates a velocity vector
    ``v <- mu*v - lr*grad`` and steps ``p <- p + v``. Momentum smooths noisy
    mini-batch gradients and accelerates along consistent directions.
    """

    def __init__(self, params_and_grads_fn, lr: float = 0.1, momentum: float = 0.0):
        self._get = params_and_grads_fn
        self.lr = lr
        self.momentum = momentum
        self._vel = {}

    def step(self) -> None:
        for i, (p, g) in enumerate(self._get()):
            if self.momentum:
                v = self._vel.get(i)
                if v is None:
                    v = np.zeros_like(p)
                v = self.momentum * v - self.lr * g
                self._vel[i] = v
                p += v
            else:
                p -= self.lr * g


class Adam:
    """Adam optimizer (Kingma & Ba, 2014), from scratch.

    Keeps two exponential moving averages per parameter:
        m <- b1*m + (1-b1)*g          (mean of gradients)
        v <- b2*v + (1-b2)*g^2        (uncentered variance)
    Bias-corrects them (they start at zero, so early steps are biased toward
    zero), then steps by ``lr * m_hat / (sqrt(v_hat) + eps)``. The per-parameter
    adaptive scaling is why Adam often "just works" without lr tuning.
    """

    def __init__(
        self,
        params_and_grads_fn,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self._get = params_and_grads_fn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = {}
        self._v = {}
        self._t = 0

    def step(self) -> None:
        self._t += 1
        b1, b2 = self.beta1, self.beta2
        for i, (p, g) in enumerate(self._get()):
            m = self._m.get(i)
            v = self._v.get(i)
            if m is None:
                m = np.zeros_like(p)
                v = np.zeros_like(p)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            self._m[i] = m
            self._v[i] = v
            m_hat = m / (1 - b1**self._t)
            v_hat = v / (1 - b2**self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
