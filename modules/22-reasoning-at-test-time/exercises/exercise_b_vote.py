"""Exercise (b): majority vote, and its accuracy averaged over every subset of n samples.

Each problem has 8 cached samples. "Vote over n" depends on which n you take;
averaging over all C(8, n) subsets removes that luck of the draw.

Predict before you run: greedy decoding scores 0.xx (see the README). Will one
sample at temperature 0.7 (n = 1) score above or below greedy? And n = 8?

Reads results/analysis.json; no model. Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_vote.py
    uv run pytest tests/test_solutions.py -k vote -v
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from common import read_results  # noqa: E402


def vote(answers: list[str | None]) -> str | None:
    """Most common answer; None does not vote; a tie goes to the answer seen first."""
    # TODO(you): count each non-None answer and note where it first appeared;
    # return the answer with the highest count, the earliest on a tie; None if no answer.
    ...


def is_right(a: str | None, gold: str) -> bool:
    return a is not None and abs(float(a) - float(gold)) < 1e-6


def vote_accuracy(problems: list[dict], n: int) -> float:
    """Mean over all C(8, n) subsets of the accuracy of vote(subset).

    Each problem is a dict with "samples" (8 extracted answers) and "gold".
    """
    subsets = list(itertools.combinations(range(8), n))
    # TODO(you): for each subset, the share of problems where vote() of those
    # samples is right; return the mean over subsets.
    ...


if __name__ == "__main__":
    a = read_results("analysis")
    print(f"greedy {a['section1']['greedy_acc']:.3f}")
    for n in range(1, 9):
        print(f"n={n}  vote {vote_accuracy(a['per_problem'], n):.3f}")
