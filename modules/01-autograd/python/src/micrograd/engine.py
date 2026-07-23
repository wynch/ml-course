"""A micrograd-style scalar autograd engine.

Every :class:`Value` is one node in a computation graph. Forward ops build the
graph and remember how to push gradient backwards through themselves via a
small closure stored in ``_backward``. Calling :meth:`Value.backward` runs a
topological sort and applies the chain rule from the output back to every leaf.

This is deliberately scalar (one number per node) so the mechanics of
reverse-mode automatic differentiation are visible with nothing hidden.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    """A single scalar value and its node in the autograd graph."""

    def __init__(
        self,
        data: float,
        _children: tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        # internal: the function that pushes our grad into our parents' grads
        self._backward: Callable[[], None] = lambda: None
        self._prev: set["Value"] = set(_children)
        self._op: str = _op
        self.label: str = label

    # ------------------------------------------------------------------ ops
    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(out)/d(self) = 1, d(out)/d(other) = 1  -> just route grad through
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # product rule: d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> "Value":
        assert isinstance(other, (int, float)), "only supports int/float powers"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            # d(x**n)/dx = n * x**(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward() -> None:
            # d(e**x)/dx = e**x = out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d(tanh(x))/dx = 1 - tanh(x)**2
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward() -> None:
            # gradient flows only where the input was positive
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------ derived operators
    def __neg__(self) -> "Value":
        return self * -1.0

    def __radd__(self, other: "Value | float") -> "Value":
        return self + other

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other: "Value | float") -> "Value":
        return (-self) + other

    def __rmul__(self, other: "Value | float") -> "Value":
        return self * other

    def __truediv__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1.0

    def __rtruediv__(self, other: "Value | float") -> "Value":
        return (self**-1.0) * other

    # --------------------------------------------------------------- backward
    def backward(self) -> None:
        """Run reverse-mode autodiff: fill ``.grad`` for every node in the graph."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)  # child appended before parent -> parents come later

        build_topo(self)

        # seed: d(self)/d(self) = 1
        self.grad = 1.0
        # walk parents-before-children by reversing topo order
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------ misc
    def __repr__(self) -> str:
        lbl = f" {self.label!r}" if self.label else ""
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}{lbl})"


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    """Return (nodes, edges) reachable from ``root`` for visualization."""
    nodes: set[Value] = set()
    edges: set[tuple[Value, Value]] = set()

    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges
