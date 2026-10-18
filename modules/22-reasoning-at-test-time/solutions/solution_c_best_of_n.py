"""Solution (c): best-of-n and a vote weighted by the model's own grade.

Each cached sample has two grades from the model, asked whether that sample
is correct: "scores", P(Yes) / (P(Yes) + P(No)), and "logodds",
log P(Yes) - log P(No). They rank samples the same way, but most scores round
to 1.0, so best-of-n ranks by log-odds; the weighted vote adds up scores. Reads results/analysis.json; no model.

    uv run python ../solutions/solution_c_best_of_n.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from common import read_results  # noqa: E402


def best_of(answers: list[str | None], scores: list[float]) -> str | None:
    """The answer of the highest-scored sample; ties go to the earliest."""
    k = max(range(len(answers)), key=lambda k: (scores[k], -k))
    return answers[k]


def weighted_vote(answers: list[str | None], scores: list[float]) -> str | None:
    """Each sample adds its score to its answer; the heaviest answer wins, ties to the one seen first."""
    w: dict[str, float] = {}
    first: dict[str, int] = {}
    for i, (a, s) in enumerate(zip(answers, scores)):
        if a is None:
            continue
        w[a] = w.get(a, 0.0) + s
        first.setdefault(a, i)
    if not w:
        return None
    return max(w, key=lambda a: (w[a], -first[a]))


def is_right(a: str | None, gold: str) -> bool:
    return a is not None and abs(float(a) - float(gold)) < 1e-6


def mean_accuracy(problems: list[dict], n: int, rule, grade: str) -> float:
    """Mean over all C(8, n) subsets of the accuracy of rule(answers, grades); grade is "scores" or "logodds"."""
    subsets = list(itertools.combinations(range(8), n))
    total = 0.0
    for sub in subsets:
        total += sum(is_right(rule([p["samples"][k] for k in sub], [p[grade][k] for k in sub]), p["gold"])
                     for p in problems) / len(problems)
    return total / len(subsets)


if __name__ == "__main__":
    a = read_results("analysis")
    for n in (1, 2, 4, 8):
        print(f"n={n}  best-of-n {mean_accuracy(a['per_problem'], n, best_of, 'logodds'):.3f}  "
              f"weighted vote {mean_accuracy(a['per_problem'], n, weighted_vote, 'scores'):.3f}")
