"""Seeded datasets for module 00a.

Every generator takes a ``seed`` and returns exactly reproducible arrays, so the
Python lane, the Zig lane, and the browser explorable can all be compared
number-for-number.

The classification data is *linearly separable by construction*: points are
drawn uniformly in a square and then rejected unless they sit at least
``margin`` away from a fixed true boundary. Turning ``margin`` down is the
knob that makes the perceptron work harder — see ``scripts/margin_sweep.py``.
"""

from __future__ import annotations

import numpy as np

# The true separator used to label the classification data. Nothing about the
# learning algorithms depends on knowing it; it is only how the labels are made.
W_TRUE = np.array([1.0, 1.6])
B_TRUE = -0.55

#: Default seed for the module's headline run (Rosenblatt's perceptron, 1958).
SEED = 1958


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def separable_2d(
    n: int = 200,
    margin: float = 0.35,
    seed: int = SEED,
    half_width: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points in ``[-hw, hw]^2`` with a guaranteed geometric margin.

    Returns ``(X, y)`` with ``X.shape == (n, 2)`` and ``y`` in ``{-1, +1}``.
    ``margin`` is the minimum Euclidean distance from any point to the true
    boundary line, so the data is separable with geometric margin ≥ ``margin``.
    """
    rng = np.random.default_rng(seed)
    u = _unit(W_TRUE)
    b_u = B_TRUE / np.linalg.norm(W_TRUE)
    keep: list[np.ndarray] = []
    while len(keep) < n:
        batch = rng.uniform(-half_width, half_width, size=(4 * n, 2))
        dist = batch @ u + b_u  # signed distance, because ||u|| == 1
        batch = batch[np.abs(dist) >= margin]
        keep.extend(batch)
    X = np.asarray(keep[:n])
    y = np.where(X @ u + b_u > 0, 1, -1).astype(int)
    return X, y


def xor_data() -> tuple[np.ndarray, np.ndarray]:
    """The four XOR corners, labels in ``{-1, +1}``. Not linearly separable."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([-1, 1, 1, -1])
    return X, y


def regression_1d(
    n: int = 40, slope: float = 1.7, intercept: float = -0.8,
    noise: float = 0.55, seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """A noisy line: ``x`` uniform in ``[-3, 3]``, ``y = slope·x + intercept + ε``."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3.0, 3.0, size=n))
    y = slope * x + intercept + rng.normal(0.0, noise, size=n)
    return x, y


def design_matrix(x: np.ndarray) -> np.ndarray:
    """``[1, x]`` — the design matrix whose column space we project onto."""
    return np.column_stack([np.ones_like(x), x])


def augment(X: np.ndarray) -> np.ndarray:
    """Append a constant 1 feature so a bias can be learned as a weight."""
    return np.column_stack([X, np.ones(len(X))])
