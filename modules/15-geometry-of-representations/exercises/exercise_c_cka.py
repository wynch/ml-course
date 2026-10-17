"""Exercise (c): linear CKA.

Two sets of activations x (n, p) and y (n, q), taken on the same n inputs.
Centre each column, then

    CKA = ||y^T x||_F^2 / (||x^T x||_F * ||y^T y||_F)

Predict before you run: which of the four pairs below give exactly 1? Which
lands near 0? Fill in the ``# TODO(you):`` line, then:

    uv run python ../exercises/exercise_c_cka.py
    uv run pytest tests/test_solutions.py -k cka -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between x (n, p) and y (n, q), rows the same n inputs."""
    # TODO(you): centre the columns of x and y, then the formula above (np.linalg.norm
    # of a matrix is its Frobenius norm).
    ...


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
