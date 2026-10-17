"""Exercise (c): BFS steps to the goal, then a ridge probe for them.

1. ``bfs_steps``: breadth-first search on a tile grid. A move goes up, down,
   left or right; walls (1), blocks (2), doors (4) and hazards (5) block it.
   Return the fewest moves to any goal cell (6), or inf.
2. ``ridge_r2``: closed-form ridge with a bias column, fit on the training
   rows, R^2 on the held-out rows.

Predict before you run: how many steps from the top-left corner of the toy
grid? And will the random encoder's R^2 sit near 0, or near the trained one?

    uv run python ../exercises/exercise_c_path_probe.py
    uv run pytest tests/test_solutions.py -k path -v
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import numpy as np

from grid_encoder import encode, load_random, load_states, load_trained, split_by_episode

GOAL = 6
BLOCKED = {1, 2, 4, 5}          # wall, block, door, hazard
CAP = 30


def bfs_steps(grid: np.ndarray, start: tuple[int, int]) -> float:
    """Fewest 4-neighbour moves from start (x, y) to any GOAL cell; inf if none is reachable."""
    h, w = grid.shape
    sx, sy = start
    seen = np.zeros((h, w), dtype=bool)
    seen[sy, sx] = True
    q = deque([(sx, sy, 0)])
    while q:
        x, y, d = q.popleft()
        # TODO(you): return d if (x, y) is a goal; otherwise push each unseen,
        # in-bounds, unblocked neighbour with d + 1. grid is indexed [y, x].
        ...
    return float("inf")


def ridge_r2(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, lam: float = 1e-2) -> float:
    """Fit w = (Xb'Xb + lam I)^-1 Xb'y on tr (Xb = X with a column of ones); R^2 on te."""
    # TODO(you): build Xb, solve for w on the training rows, predict the
    # held-out rows, return 1 - SS_res / SS_tot (add 1e-12 to SS_tot).
    ...


def main() -> None:
    toy = np.array([[0, 0, 1, 6],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])
    print(f"toy grid, from (0, 0): {bfs_steps(toy, (0, 0))} steps")

    st = load_states()
    raw = st["goal_dist"].astype(np.float64)
    y = np.where(np.isfinite(raw), np.minimum(raw, CAP), CAP).astype(np.float32)[:, None]
    tr, te = split_by_episode(st["ep"], seed=0)
    for name, enc in (("trained", load_trained()), ("random_0", load_random(0))):
        Z = encode(enc, st["windows"], st["scalars"])
        print(f"{name}: path-length R^2 = {ridge_r2(Z, y, tr, te):.4f}")


if __name__ == "__main__":
    main()
