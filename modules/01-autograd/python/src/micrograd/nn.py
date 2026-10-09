"""A tiny neural network built entirely out of :class:`Value` scalars.

Neuron -> Layer -> MLP, exactly like Karpathy's micrograd. Because each weight
and bias is a ``Value``, calling ``loss.backward()`` differentiates the whole
network with no extra machinery.
"""

from __future__ import annotations

import random
from typing import List

from .engine import Value


class Module:
    """Base class: gives every model a ``parameters()`` list and ``zero_grad()``."""

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> List[Value]:
        return []


class Neuron(Module):
    """A single neuron: w . x + b, optionally passed through a nonlinearity."""

    def __init__(self, nin: int, nonlin: str = "tanh") -> None:
        self.w: List[Value] = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b: Value = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: List[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == "tanh":
            return act.tanh()
        if self.nonlin == "relu":
            return act.relu()
        return act  # linear

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{self.nonlin or 'linear'}Neuron({len(self.w)})"


class Layer(Module):
    """A fully-connected layer: a list of neurons over the same inputs."""

    def __init__(self, nin: int, nout: int, nonlin: str = "tanh") -> None:
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x: List[Value]):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer[{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A multi-layer perceptron. Hidden layers use ``nonlin``; output is linear."""

    def __init__(self, nin: int, nouts: List[int], nonlin: str = "tanh") -> None:
        sizes = [nin] + nouts
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                nonlin=nonlin if i < len(nouts) - 1 else "linear",
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x: List[Value]):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP[{', '.join(str(l) for l in self.layers)}]"
