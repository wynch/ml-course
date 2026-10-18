"""Solution (b): majority vote, and its accuracy averaged over every subset of n samples.

Reads the cached answers in results/analysis.json; no model.

    uv run python ../solutions/solution_b_vote.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from common import read_results  # noqa: E402


def vote(answers: list[str | None]) -> str | None:
    """Most common answer; None does not vote; a tie goes to the answer seen first."""
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for i, a in enumerate(answers):
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
        first.setdefault(a, i)
    if not counts:
        return None
    return max(counts, key=lambda a: (counts[a], -first[a]))


def is_right(a: str | None, gold: str) -> bool:
    return a is not None and abs(float(a) - float(gold)) < 1e-6


def vote_accuracy(problems: list[dict], n: int) -> float:
    """Mean over all C(8, n) subsets of the accuracy of vote(subset)."""
    subsets = list(itertools.combinations(range(8), n))
    total = 0.0
    for sub in subsets:
        total += sum(is_right(vote([p["samples"][k] for k in sub]), p["gold"]) for p in problems) / len(problems)
    return total / len(subsets)


if __name__ == "__main__":
    a = read_results("analysis")
    print(f"greedy {a['section1']['greedy_acc']:.3f}")
    for n in range(1, 9):
        print(f"n={n}  vote {vote_accuracy(a['per_problem'], n):.3f}")
