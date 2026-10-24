"""Exercise (c): the best-so-far curve, and how much of the budget the winner needed.

Each run logs one row per instruction it scored on all 60 training questions:
"calls" is the budget spent once that pass is done, "acc" its training
accuracy. The curve the README plots is the best accuracy seen so far
against the budget spent.

Predict before you run: the budget is 2,400 questions scored. Does the best
instruction of a run tend to turn up in its first quarter, or near the end?

Reads results/runs/*.jsonl; no model. Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_c_curve.py
    uv run pytest tests/test_solutions.py -k curve -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: E402


def best_so_far(rows: list[dict]) -> list[tuple[int, float]]:
    """[(budget spent after the pass, best accuracy so far)], one per "full" row."""
    # TODO(you): walk the rows in order; skip rows whose "event" is not "full";
    #            keep a running maximum of "acc".
    ...


def budget_to_best(rows: list[dict]) -> int:
    """Budget spent when the final best accuracy was first reached."""
    # TODO(you): from best_so_far(rows), the first point whose accuracy equals the last one.
    ...


if __name__ == "__main__":
    for path in sorted(common.RUNS.glob("*.jsonl")):
        rows = common.read_jsonl(path)
        print(f"{path.stem:12s} best {best_so_far(rows)[-1][1]:.3f} "
              f"reached at {budget_to_best(rows)} of 2400")
