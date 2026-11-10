"""Exercise 4 — the moving-average update, and what the batch and the decay do.

Everything around the update is written for you: the shuffle, the batches, the
shared assignment step, the exact final pass. Fill in the two `# TODO(you):`
blocks: the cosine step, and the two lines that move the running count and
sum. The check first holds your update against the module's, then runs four
settings over 100 starts on the gap-5 data of Part 5.

Before you run it, write down a guess for each row of the table: Lloyd ends
stuck on 55% of starts at this gap and the module's setting on about a third.
Does a constant step get stuck more or less often? A batch of 64? Of 2,048?

Run:  cd python && uv run ../exercises/ex4_moving_average.py
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
            # TODO(you): the step for update t of total.
            #   cosine=True  -> alpha0 * (0.1 + 0.9 * (1 + cos(pi * t / total)) / 2),
            #                   alpha0 at the first update, alpha0 / 10 at the last
            #   cosine=False -> alpha0 every time
            # alpha = ???

            # TODO(you): move the running count N (k,) and sum S (k, d) a step
            # alpha toward this batch's count n_j and sum s_j.
            # N = ???
            # S = ???
            raise NotImplementedError("write the step and the update")
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
          "FAIL — your centres differ from the module's.")
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
    print("\nCompare the table with your guesses. Why does the batch size matter so\n"
          "much more here than the final error of the starts that escape?")
    return ok


if __name__ == "__main__":
    _check()
