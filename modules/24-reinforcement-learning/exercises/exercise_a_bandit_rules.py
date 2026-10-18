"""Exercise (a): UCB1 and Thompson sampling, written for 20 bandits at once.

``counts`` and ``sums`` have shape (bandits, arms): how often each arm was
pulled and how many of those pulls paid 1. ``t`` counts pulls so far, from 1.
Return one arm index per bandit. Break ties with ``argmax_random``.

Predict before you run: over 2,000 pulls of the module's five arms
(0.2, 0.4, 0.5, 0.6, 0.7), which rule ends with the lower median regret, and
by about how much? Over 10,000 pulls the module measured 228 for UCB1 and 43
for Thompson. Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_bandit_rules.py
    uv run pytest tests/test_solutions.py -k bandit -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from bandits import MEANS, argmax_random


def ucb_choose(counts, sums, t, rng):
    """UCB1: pull arm t-1 for the first k pulls; then argmax of mean + sqrt(2 ln t / n)."""
    n, k = counts.shape
    # TODO(you): if t <= k, return np.full(n, t - 1).
    # TODO(you): otherwise mean = sums / counts, bonus = sqrt(2 ln t / counts); return argmax_random(mean + bonus, rng).
    ...


def thompson_choose(counts, sums, t, rng):
    """Thompson sampling, Beta(1, 1) prior: one Beta(1 + wins, 1 + losses) draw per arm, pull the best draw."""
    # TODO(you): draws = rng.beta(1 + sums, 1 + counts - sums); return argmax_random(draws, rng).
    ...


def regret(choose, pulls=2000, seeds=20):
    """Median pseudo-regret over ``seeds`` bandits after ``pulls`` pulls."""
    data = np.random.default_rng(0)
    probs = np.stack([data.permutation(MEANS) for _ in range(seeds)])
    rng = np.random.default_rng(1)
    counts, sums, reg = np.zeros_like(probs), np.zeros_like(probs), np.zeros(seeds)
    rows = np.arange(seeds)
    for t in range(1, pulls + 1):
        a = choose(counts, sums, t, rng)
        r = (data.random(seeds) < probs[rows, a]).astype(float)
        counts[rows, a] += 1
        sums[rows, a] += r
        reg += probs.max(1) - probs[rows, a]
    return float(np.median(reg))


def main() -> None:
    for name, fn in (("UCB1", ucb_choose), ("Thompson", thompson_choose)):
        print(f"{name:9s} median regret after 2,000 pulls: {regret(fn):.1f}")


if __name__ == "__main__":
    main()
