"""Exercise (b): count the linear regions along a segment, exactly.

A ReLU net with two hidden layers, weights [(W1, b1), (W2, b2), ...], a layer
being x @ W + b. Along x(t) = (1 - t) a + t b, layer 1's pre-activations are
linear in t, so each changes sign at most once, at a point you can solve for.
Between two such points layer 1 is a fixed linear map, so layer 2's
pre-activations are linear in t too, and each changes sign at most once per piece.

1. ``crossings``: the t in (lo, hi) where a linear pre-activation changes sign.
2. ``count_on_segment``: layer-1 breaks, then layer-2 breaks inside each piece;
   the count is the number of distinct breaks plus one.

Predict before you run: at init, how does the count per segment change when the
width doubles from 64 to 128? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_regions.py
    uv run pytest tests/test_solutions.py -k regions -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import numpy as np
import regions


def crossings(pa: np.ndarray, pb: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Where a pre-activation, linear in t from pa at t = lo to pb at t = hi, changes sign."""
    # TODO(you): keep the units whose sign differs at the two ends; solve for the zero;
    # return only the t strictly between lo and hi.
    ...


def count_on_segment(weights, a: np.ndarray, b: np.ndarray) -> int:
    """Regions of a ReLU net with two hidden layers met by the segment a -> b."""
    (w1, b1), (w2, b2) = weights[0], weights[1]
    ha, hb = a @ w1 + b1, b @ w1 + b1                      # layer 1 at the two ends
    # TODO(you): t1 = layer-1 crossings; for each piece between them, the layer-1 mask
    # at its middle, layer 2's pre-activation at the piece's two ends, its crossings.
    ...


def main() -> None:
    rng = np.random.default_rng(0)
    for w in (32, 64, 128, 256):
        weights = regions.net_weights(regions.fashion_net(w))
        counts = [count_on_segment(weights, rng.random(784), rng.random(784)) for _ in range(20)]
        print(f"784-{w}-{w}-10 at init: {np.mean(counts):.1f} regions per random segment")


if __name__ == "__main__":
    main()
