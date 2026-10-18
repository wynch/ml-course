"""Exercise (a): three classic rules for a prompt of (x, y) pairs, and their error by n.

Each rule sees the n pairs of one prompt (X is n x d, y has n entries) and returns a
weight vector w; the guess for a query x is w . x.

1. ``least_squares``: the minimum-norm solution, so it is defined when n < d.
2. ``ridge``: least squares with a penalty lam * |w|^2.
3. ``one_gd_step``: start at w = 0 and take one gradient step on
   (1/2) sum (w.x_i - y_i)^2, with step size 1/(n + d + 1).

Predict before you run: with d = 5 and no noise, which rule wins at n = 3? At n = 10,
how far is one gradient step from least squares? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_rules.py
    uv run pytest tests/test_solutions.py -k rules -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np


def least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimum-norm least squares: the pseudo-inverse, defined for n < d too."""
    # TODO(you): one line (np.linalg.pinv or np.linalg.lstsq).
    ...


def ridge(X: np.ndarray, y: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """(X^T X + lam I)^-1 X^T y."""
    # TODO(you): one line with np.linalg.solve.
    ...


def one_gd_step(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One gradient step from w = 0 on (1/2) sum (w.x_i - y_i)^2, step 1/(n + d + 1)."""
    n, d = X.shape
    # TODO(you): the gradient of the loss at w = 0, then one step against it.
    ...


def mse_over_d(rule, d: int = 5, n: int = 10, prompts: int = 500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(prompts):
        w = rng.standard_normal(d)
        X, xq = rng.standard_normal((n, d)), rng.standard_normal(d)
        err += (xq @ rule(X, X @ w) - xq @ w) ** 2
    return err / prompts / d


def main() -> None:
    print("MSE/d, d = 5, 500 prompts")
    print("  n   least squares   ridge 0.1   one GD step   (d+1)/(n+d+1)")
    for n in (3, 5, 10, 20):
        print(f"  {n:<3} {mse_over_d(least_squares, n=n):13.4f} {mse_over_d(ridge, n=n):11.4f} "
              f"{mse_over_d(one_gd_step, n=n):13.4f} {6 / (n + 6):15.4f}")


if __name__ == "__main__":
    main()
