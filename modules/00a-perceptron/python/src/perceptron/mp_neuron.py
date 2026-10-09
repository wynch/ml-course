"""McCulloch–Pitts neurons: logic with a threshold, no learning at all.

A 1943-style unit fires when a weighted sum of binary inputs reaches a
threshold. The weights are *chosen by hand* — that is the whole point. This
file is what the perceptron replaces: the same architecture, plus a rule for
finding the weights yourself.
"""

from __future__ import annotations

from collections.abc import Sequence

# name -> (weights, threshold). Fires when  w·x >= theta.
GATES: dict[str, tuple[tuple[float, ...], float]] = {
    "AND": ((1.0, 1.0), 2.0),
    "OR": ((1.0, 1.0), 1.0),
    "NAND": ((-1.0, -1.0), -1.0),
    "NOR": ((-1.0, -1.0), 0.0),
    "NOT": ((-1.0,), 0.0),
    "MAJORITY3": ((1.0, 1.0, 1.0), 2.0),
}

#: The four binary input pairs, in a fixed order used by every truth table here.
INPUTS2 = ((0, 0), (0, 1), (1, 0), (1, 1))


def fire(weights: Sequence[float], theta: float, x: Sequence[float]) -> int:
    """1 if ``w·x >= theta`` else 0 — the whole McCulloch–Pitts model."""
    total = sum(w * xi for w, xi in zip(weights, x))
    return 1 if total >= theta else 0


def gate(name: str, x: Sequence[float]) -> int:
    """Evaluate a named gate from :data:`GATES`."""
    weights, theta = GATES[name]
    return fire(weights, theta, x)


def truth_table(name: str, inputs: Sequence[Sequence[int]] = INPUTS2) -> list[tuple]:
    """``[(x1, x2, out), ...]`` for a named gate."""
    return [tuple(x) + (gate(name, x),) for x in inputs]


def xor_from_gates(x: Sequence[float]) -> int:
    """XOR built from three MP neurons — two in a hidden layer, one on top.

    ``XOR(a, b) = AND(OR(a, b), NAND(a, b))``. No single threshold unit can do
    this; two layers can. That is the whole 1969 story in one line, and the
    reason module 02 exists.
    """
    h1 = gate("OR", x)
    h2 = gate("NAND", x)
    return gate("AND", (h1, h2))
