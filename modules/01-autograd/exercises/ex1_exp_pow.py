"""Exercise 1 — implement the backward closures for exp and pow.

This file has a small self-contained autograd engine. The forward pass for
`exp` and `__pow__` is written for you, but their `_backward` closures are
missing. Fill in the two `# TODO(you):` blocks so that gradients are correct.

A numeric gradient check runs at the bottom. It will FAIL until you finish.

Run:  cd python && uv run ../exercises/ex1_exp_pow.py
  (or, if you have a system python with nothing needed: python3 exercises/ex1_exp_pow.py)
"""

import math


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

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        # TODO(you): set out._backward so that self.grad gets the right update.
        # Hint: d(e**x)/dx = e**x, and e**x is exactly out.data.
        # Remember to ACCUMULATE (+=) and multiply by out.grad (the chain rule).
        #
        # def _backward():
        #     self.grad += ???
        # out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        # TODO(you): set out._backward.
        # Hint: d(x**n)/dx = n * x**(n-1). Multiply by out.grad and ACCUMULATE.
        #
        # def _backward():
        #     self.grad += ???
        # out._backward = _backward

        return out

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


def _numeric_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def _check():
    ok = True

    # check exp: g = exp(a) * a  (mixes exp with mul)
    a = Value(0.7)
    g = a.exp() * a
    g.backward()
    num = _numeric_grad(lambda t: math.exp(t) * t, 0.7)
    print(f"exp: analytic={a.grad:+.6f}  numeric={num:+.6f}  err={abs(a.grad-num):.1e}")
    ok = ok and abs(a.grad - num) < 1e-4

    # check pow: h = b**3
    b = Value(1.5)
    h = b ** 3
    h.backward()
    num = _numeric_grad(lambda t: t ** 3, 1.5)
    print(f"pow: analytic={b.grad:+.6f}  numeric={num:+.6f}  err={abs(b.grad-num):.1e}")
    ok = ok and abs(b.grad - num) < 1e-4

    print("\n" + ("PASS — nice, your backward closures are correct!" if ok
                  else "FAIL — implement the two TODO blocks above."))
    return ok


if __name__ == "__main__":
    _check()
