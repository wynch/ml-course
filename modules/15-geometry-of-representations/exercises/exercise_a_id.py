"""Exercise (a): two intrinsic-dimension estimators from neighbour distances.

``d`` holds, for each point, the sorted distances to its 20 nearest neighbours
(``idest.knn_dists`` computes it). Write:

1. ``twonn``: mu = r2 / r1 per point; sort; drop the largest 10%; fit
   -log(1 - i/n) = d * log(mu_i) through the origin; return the slope.
2. ``mle_k``: Levina and Bickel at one k: 1 / mean of log(T_k / T_j), j = 1..k-1,
   the mean taken over points and j together.

Predict before you run: on a curved 10-d cube with 5,000 points, will the two
estimates land above 10, at 10, or below? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_id.py
    uv run pytest tests/test_solutions.py -k id -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import idest
import numpy as np


def twonn(d: np.ndarray) -> float:
    """d (n, >= 2): sorted distances to each point's nearest neighbours. Returns TwoNN's d."""
    # TODO(you): mu = d[:, 1] / d[:, 0], sorted; keep the smallest 90%;
    # x = log(mu), y = -log(1 - i/n) for i = 1..keep; return (x @ y) / (x @ x).
    ...


def mle_k(d: np.ndarray, k: int) -> float:
    """Levina-Bickel at one k: 1 / mean over points and j < k of log(T_k / T_j)."""
    # TODO(you): T_k is column k-1, T_j the columns before it.
    ...


def main() -> None:
    for dim in (2, 5, 10):
        x = idest.curved_cube(dim, 5000, seed=dim)
        d = idest.knn_dists(x, 20)
        mle = np.mean([mle_k(d, k) for k in range(10, 21)])
        print(f"curved cube, true dimension {dim}: TwoNN {twonn(d):.2f}, MLE {mle:.2f}")


if __name__ == "__main__":
    main()
