"""Exercise (a): Bradley-Terry by hand.

Twelve items have hidden scores s. A judge prefers i over j with probability
sigmoid(s_i - s_j). From (winner, loser) pairs alone, recover s.

1. ``bt_grad``: the gradient of the mean loss -log sigmoid(s_w - s_l) with
   respect to every score. For one pair, the winner's score gets
   -(1 - sigmoid(s_w - s_l)) and the loser's gets the opposite.
2. ``fit``: gradient descent from zero. Only differences between scores are
   identified, so subtract the mean after every step.

Predict before you run: with 30 comparisons among 12 items, is Kendall's tau
between the hidden and fitted scores above or below 0.8? And with 1,000?

    uv run python ../exercises/exercise_a_bt.py
    uv run pytest tests/test_solutions.py -k bt -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from bt_synthetic import kendall_tau, simulate, true_scores


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bt_grad(s: np.ndarray, w: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Gradient of mean(-log sigmoid(s[w] - s[l])) with respect to s. w, l: index arrays."""
    # TODO(you): one value per pair, then add it into the winners' entries
    # and subtract it from the losers' (np.add.at handles repeated indices).
    ...


def fit(w: np.ndarray, l: np.ndarray, n_items: int, steps: int = 4000, lr: float = 4.0) -> np.ndarray:
    s = np.zeros(n_items)
    for _ in range(steps):
        # TODO(you): a gradient step (add 1e-4 * s to the gradient, so an item
        # that never lost stays finite), then centre s.
        ...
    return s


def main() -> None:
    s_true = true_scores()
    for n in (30, 1000):
        w, l = simulate(s_true, n, seed=7)
        s = fit(w, l, len(s_true))
        print(f"{n:>5} comparisons: Kendall tau {kendall_tau(s_true, s):.3f}")


if __name__ == "__main__":
    main()
