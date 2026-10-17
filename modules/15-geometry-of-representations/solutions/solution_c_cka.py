"""Reference solution, exercise (c): linear CKA."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between x (n, p) and y (n, q), rows the same n inputs."""
    xc, yc = x - x.mean(0), y - y.mean(0)                  # centre each column
    return float(np.linalg.norm(yc.T @ xc) ** 2 / (np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc)))


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 50)) @ rng.normal(size=(50, 50))
    q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
    a = rng.normal(size=(50, 50))                          # invertible, but not a rotation
    print(f"x vs 5 x Q (rotation, scale): {linear_cka(x, 5 * x @ q):.4f}")
    print(f"x vs x A (a general linear map): {linear_cka(x, x @ a):.4f}")
    print(f"x vs fresh noise:             {linear_cka(x, rng.normal(size=x.shape)):.4f}")
    print(f"x vs ReLU(x):                 {linear_cka(x, np.maximum(x, 0)):.4f}")


if __name__ == "__main__":
    main()
