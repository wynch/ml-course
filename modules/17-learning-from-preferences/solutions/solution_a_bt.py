"""Solution (a): Bradley-Terry by hand. See exercises/exercise_a_bt.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from bt_synthetic import kendall_tau, simulate, true_scores


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bt_grad(s: np.ndarray, w: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Gradient of mean(-log sigmoid(s[w] - s[l])) with respect to s."""
    g_pair = -(1.0 - sigmoid(s[w] - s[l])) / len(w)
    g = np.zeros_like(s)
    np.add.at(g, w, g_pair)
    np.add.at(g, l, -g_pair)
    return g


def fit(w: np.ndarray, l: np.ndarray, n_items: int, steps: int = 4000, lr: float = 4.0) -> np.ndarray:
    s = np.zeros(n_items)
    for _ in range(steps):
        s -= lr * (bt_grad(s, w, l) + 1e-4 * s)
        s -= s.mean()
    return s


def main() -> None:
    s_true = true_scores()
    for n in (30, 1000):
        w, l = simulate(s_true, n, seed=7)
        s = fit(w, l, len(s_true))
        print(f"{n:>5} comparisons: Kendall tau {kendall_tau(s_true, s):.3f}")


if __name__ == "__main__":
    main()
