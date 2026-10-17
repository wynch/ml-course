"""Reference solution, exercise (a): TwoNN and MLE from neighbour distances."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import idest
import numpy as np


def twonn(d: np.ndarray) -> float:
    """d (n, >= 2): sorted distances to each point's nearest neighbours. Returns TwoNN's d."""
    mu = np.sort(d[:, 1] / d[:, 0])                        # r2 / r1 per point, ascending
    n = len(mu)
    keep = int(np.floor(n * 0.9))                          # drop the largest 10%
    xs = np.log(mu[:keep])
    ys = -np.log(1 - np.arange(1, keep + 1) / n)           # -log(1 - empirical CDF)
    return float((xs @ ys) / (xs @ xs))                    # slope of a fit through the origin


def mle_k(d: np.ndarray, k: int) -> float:
    """Levina-Bickel at one k: 1 / mean over points and j < k of log(T_k / T_j)."""
    return float(1.0 / np.log(d[:, k - 1:k] / d[:, : k - 1]).mean())


def main() -> None:
    for dim in (2, 5, 10):
        x = idest.curved_cube(dim, 5000, seed=dim)
        d = idest.knn_dists(x, 20)
        mle = np.mean([mle_k(d, k) for k in range(10, 21)])
        print(f"curved cube, true dimension {dim}: TwoNN {twonn(d):.2f}, MLE {mle:.2f}")


if __name__ == "__main__":
    main()
