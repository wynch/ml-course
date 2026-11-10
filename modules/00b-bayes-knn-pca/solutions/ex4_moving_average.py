"""Solution 4 — the moving-average update, and what the batch and the decay do.

The step is the cosine of eq. 5 in the paper; the update is two lines, one for
the running count and one for the running sum. The lesson is in the table: on
2-D data, noise is what gets a centre out of the trap. A smaller batch, or a
step that never decays, gives more noise and fewer stuck starts; a batch of
2,048 gives so little that the rule ends stuck about as often as Lloyd.
Every row keeps alpha0 = 0.8, so a smaller batch here also means a larger step
per point; the paper's step law would set alpha0 near 0.23 for batch 64.

Run:  cd python && uv run ../solutions/ex4_moving_average.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from origins.data import two_gaussians
from origins.kmeans import (
    assign,
    batch_stats,
    centre_means,
    init_from_data,
    kmeans_error,
    moving_average_kmeans,
)

GAP, K, PASSES, STARTS, STUCK = 5.0, 4, 10, 100, 0.02
VARIANTS = [  # (name, batch, alpha0, cosine)
    ("batch 256, cosine decay", 256, 0.8, True),
    ("batch 256, constant step", 256, 0.8, False),
    ("batch 64, cosine decay", 64, 0.8, True),
    ("batch 2048, cosine decay", 2048, 0.8, True),
]


def my_moving_average(X, C0, passes, batch, alpha0, seed=0, cosine=True):
    """Return the centres after `passes` passes and one exact final pass."""
    C = np.array(C0, dtype=np.float64, copy=True)
    n, k = len(X), len(C)
    steps = n // batch
    total = passes * steps
    N = np.full(k, 1e-10)
    S = N[:, None] * C
    rng = np.random.default_rng(seed)
    t = 0
    for _ in range(passes):
        order = rng.permutation(n)
        for b in range(steps):
            xb = X[order[b * batch : (b + 1) * batch]]
            labels, _ = assign(xb, C)
            n_j, s_j = batch_stats(xb, labels, k)
            if cosine:
                alpha = alpha0 * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * t / total)))
            else:
                alpha = alpha0
            N = (1.0 - alpha) * N + alpha * n_j
            S = (1.0 - alpha) * S + alpha * s_j
            live = N > 0
            C[live] = S[live] / N[live, None]
            t += 1
    labels, _ = assign(X, C)
    return centre_means(X, labels, C)


def _check():
    X, _ = two_gaussians(10_000, sep=GAP, sigma=1.0, seed=5)
    C0 = init_from_data(X, K, seed=3)
    ref, _ = moving_average_kmeans(X, C0, PASSES, batch=256, alpha0=0.8, seed=1003)
    ok = np.allclose(my_moving_average(X, C0, PASSES, 256, 0.8, seed=1003), ref, atol=1e-9)
    print("PASS — your update matches the module's." if ok else
          "FAIL — fill in the two TODO blocks.")
    if not ok:
        return False

    finals = {}
    for name, batch, a0, cos in VARIANTS:
        finals[name] = np.array([
            kmeans_error(X, my_moving_average(X, init_from_data(X, K, seed=s), PASSES, batch,
                                              a0, seed=1000 + s, cosine=cos))
            for s in range(STARTS)])
    best = min(v.min() for v in finals.values())
    print(f"\nGap {GAP:.0f}, k = {K}, {STARTS} starts; best final error {best:.4f}")
    print("  variant                      stuck (> 2% above best)   median error of the rest")
    for name, v in finals.items():
        good = v[v <= (1 + STUCK) * best]
        print(f"  {name:27s}  {np.mean(v > (1 + STUCK) * best):6.0%}"
              f"                    {np.median(good):.4f}")
    print("\nNoise gets a centre out of the trap: a smaller batch or a step that never\n"
          "decays gives more of it and fewer stuck starts; batch 2048 gives so little\n"
          "that the rule ends stuck about as often as Lloyd (55% at this gap). The\n"
          "constant step pays for its escapes: the starts that do not get stuck end\n"
          "a little higher than with the decay, which lets centres settle.\n"
          "Every row keeps alpha0 = 0.8, so the small batch also takes a larger\n"
          "step per point; try the law's alpha0 for batch 64, about 0.23.")
    return ok


if __name__ == "__main__":
    _check()
