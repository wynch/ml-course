"""Reference solution, exercise (b): count the linear regions along a segment, exactly."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import numpy as np
import regions


def crossings(pa: np.ndarray, pb: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Where a pre-activation, linear in t from pa at t = lo to pb at t = hi, changes sign."""
    c = (pa > 0) != (pb > 0)
    t = lo + (hi - lo) * pa[c] / (pa[c] - pb[c])
    return t[(t > lo) & (t < hi)]


def count_on_segment(weights, a: np.ndarray, b: np.ndarray) -> int:
    """Regions of a ReLU net with two hidden layers met by the segment a -> b."""
    (w1, b1), (w2, b2) = weights[0], weights[1]
    ha, hb = a @ w1 + b1, b @ w1 + b1                      # layer 1 at the two ends
    t1 = np.unique(crossings(ha, hb))
    edges = np.concatenate([[0.0], t1, [1.0]])
    breaks = list(t1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = (lo + hi) / 2
        mask = ((1 - mid) * ha + mid * hb) > 0             # layer-1 pattern on this piece
        pre2 = lambda t: (((1 - t) * ha + t * hb) * mask) @ w2 + b2
        breaks.extend(crossings(pre2(lo), pre2(hi), lo, hi))
    return len(np.unique(breaks)) + 1


def main() -> None:
    rng = np.random.default_rng(0)
    for w in (32, 64, 128, 256):
        weights = regions.net_weights(regions.fashion_net(w))
        counts = [count_on_segment(weights, rng.random(784), rng.random(784)) for _ in range(20)]
        print(f"784-{w}-{w}-10 at init: {np.mean(counts):.1f} regions per random segment")


if __name__ == "__main__":
    main()
