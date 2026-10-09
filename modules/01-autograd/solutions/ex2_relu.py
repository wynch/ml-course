"""Solution 2 — ReLU implemented and used to retrain the MLP.

Run:  cd python && uv run ../solutions/ex2_relu.py
"""

import math
import random

import numpy as np
from sklearn.datasets import make_moons

NONLIN = "relu"  # switched from "tanh" -> "relu"


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            # gradient flows only where the input was positive
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def nonlin(v):
    return v.relu() if NONLIN == "relu" else v.tanh()


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            layer = []
            for _ in range(sizes[i + 1]):
                w = [Value(random.uniform(-1, 1)) for _ in range(sizes[i])]
                b = Value(0.0)
                layer.append((w, b, i < len(nouts) - 1))
            self.layers.append(layer)

    def __call__(self, x):
        for layer in self.layers:
            out = []
            for w, b, use_nl in layer:
                act = sum((wi * xi for wi, xi in zip(w, x)), b)
                out.append(nonlin(act) if use_nl else act)
            x = out
        return x[0]

    def parameters(self):
        return [p for layer in self.layers for (w, b, _) in layer for p in w + [b]]


def main():
    random.seed(1337)
    np.random.seed(1337)
    X, y = make_moons(n_samples=100, noise=0.1, random_state=1337)
    y = y * 2 - 1

    model = MLP(2, [16, 16, 1])
    for k in range(60):
        losses = [(1 + -yi * model([Value(a), Value(b)])).relu()
                  for (a, b), yi in zip(X, y)]
        loss = sum(losses) * (1.0 / len(losses))
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        lr = 1.0 - 0.9 * k / 60
        for p in model.parameters():
            p.data -= lr * p.grad
        if k % 10 == 0 or k == 59:
            acc = np.mean([(yi > 0) == (model([Value(a), Value(b)]).data > 0)
                           for (a, b), yi in zip(X, y)])
            print(f"[{NONLIN}] step {k:3d}  loss {loss.data:.4f}  acc {acc*100:.0f}%")


if __name__ == "__main__":
    main()
