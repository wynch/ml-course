"""Build a small expression, run backward, and draw the labelled graph.

The expression is a single tanh neuron:  n = tanh(x1*w1 + x2*w2 + b)
After backward(), every node carries both its forward ``data`` and its
``grad`` = d(n)/d(node). Run:  uv run scripts/computation_graph.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

from micrograd.engine import Value
from micrograd.draw import draw_graph


def main() -> None:
    # inputs
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bias
    b = Value(6.8813735870195432, label="b")

    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1w1+x2w2"
    n = x1w1x2w2 + b
    n.label = "n"
    o = n.tanh()
    o.label = "o = tanh(n)"

    o.backward()

    print("forward + backward on o = tanh(x1*w1 + x2*w2 + b)")
    for v in (x1, w1, x2, w2, b, n, o):
        print(f"  {v.label:12s} data={v.data:+.4f}  grad={v.grad:+.4f}")

    out = draw_graph(o, str(FIGDIR / "computation_graph"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
