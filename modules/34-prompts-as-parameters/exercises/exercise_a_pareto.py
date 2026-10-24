"""Exercise (a): the Pareto front that GEPA-style search draws its parents from.

Each candidate instruction has a 0/1 row: right or wrong on each of the 60
training questions. A candidate enters the front when it is best on at least
one question and no other candidate dominates it (as good everywhere, better
somewhere). It is then drawn as a parent with weight equal to the number of
questions it is best on.

Predict before you run: on the matrix below, which candidates enter the
front? Candidate 0 has the best average. Is it the only parent?

No model. Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_pareto.py
    uv run pytest tests/test_solutions.py -k pareto -v
"""

from __future__ import annotations

import numpy as np


def pareto_parents(rows: list[list[int]]) -> dict[int, int]:
    """{candidate index: questions it is best on}, for undominated candidates only.

    Questions that no candidate gets right count for no one. If no candidate
    is left, return the best average alone with weight 1 (earliest on ties).
    """
    R = np.array(rows)
    # TODO(you): the best score on each question (a column maximum).
    # TODO(you): for each candidate, how many questions it ties the best on,
    #            counting only questions where the best is 1.
    # TODO(you): drop candidates with no such question, and dominated ones.
    ...


if __name__ == "__main__":
    rows = [[1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]]
    print(pareto_parents(rows))
