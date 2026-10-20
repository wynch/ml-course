"""Solution (b): pass@k from n samples, and the base model against the RL models.

pass@k is the chance that at least one of k tries is right. With n >= k
samples of which c are right, the unbiased estimate (Chen et al., 2021) is
1 - C(n - c, k) / C(n, k): the chance that k samples drawn from the n,
without replacement, are all wrong, subtracted from 1.

Predict first: the base model is right on 59% of single samples. At k = 16,
is it nearer 60% or 100%? Does the RL model ever fall below it?

    uv run python ../solutions/solution_b_pass_at_k.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curve(right_per_problem: list[int], n: int, ks: list[int]) -> list[float]:
    return [sum(pass_at_k(n, c, k) for c in right_per_problem) / len(right_per_problem) for k in ks]


def main() -> None:
    ks = [1, 2, 4, 8, 16, 32, 64, 128]
    print("k      " + "".join(f"{k:>8}" for k in ks))
    for name in ["base_eval", "grpo_s0", "drgrpo_s0", "rloo_s0", "random_s0", "random_clip_s0"]:
        f = RUNS / f"{name}.json"
        if not f.exists():
            continue
        ev = json.loads(f.read_text())["final"]
        c = curve(ev["right_per_problem"], ev["n_samples"], ks)
        print(f"{name:<15}" + "".join(f"{100 * x:8.2f}" for x in c))


if __name__ == "__main__":
    main()
