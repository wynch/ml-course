"""Exercise (b): is an instruction's gain on the test set more than luck?

Two instructions answer the same 500 test questions. Only the questions where
they disagree carry information: b right and a wrong (a win for b), or the
reverse (a loss). If the two were equally good, each disagreement would be a
coin flip. The exact two-sided sign test asks how likely a split at least
this uneven is under that coin.

Predict before you run: the instruction we wrote by hand beats the baseline
by 7 points on test (56.0% against 49.0%). Is p below 0.01?

Reads results/baseline.json; no model. Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_sign_test.py
    uv run pytest tests/test_solutions.py -k sign -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from common import read_results  # noqa: E402


def sign_test(a: list[int], b: list[int]) -> dict:
    """{"wins": b right and a wrong, "losses": the reverse, "p": two-sided exact p}."""
    # TODO(you): count wins and losses over paired entries.
    # TODO(you): n = wins + losses, k = the smaller of the two;
    #            p = 2 * P(Binomial(n, 1/2) <= k), capped at 1; p = 1 when n = 0.
    ...


if __name__ == "__main__":
    base = read_results("baseline")
    print(sign_test(base["baseline"]["test_right"], base["by_hand"]["test_right"]))
