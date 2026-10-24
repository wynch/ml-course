"""Solution (a): the Pareto front that GEPA-style search draws its parents from."""

from __future__ import annotations

import numpy as np


def pareto_parents(rows: list[list[int]]) -> dict[int, int]:
    """{candidate index: questions it is best on}, for undominated candidates only."""
    R = np.array(rows)
    best = R.max(axis=0)
    wins = {}
    for j in range(len(R)):
        n = int(((R[j] == best) & (best > 0)).sum())
        if n == 0:
            continue
        dominated = any((R[k] >= R[j]).all() and (R[k] > R[j]).any()
                        for k in range(len(R)) if k != j)
        if not dominated:
            wins[j] = n
    if not wins:
        wins = {int(R.mean(axis=1).argmax()): 1}
    return wins


if __name__ == "__main__":
    rows = [[1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]]
    print(pareto_parents(rows))   # {0: 4, 1: 2}: 2 and 3 are dominated by 0
