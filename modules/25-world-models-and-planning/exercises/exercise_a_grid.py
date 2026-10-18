"""Exercise (a): the grid world's two rules.

1. ``step``: move the agent one cell; a move into a wall or off the grid
   leaves it where it was.
2. ``bfs``: the number of steps from every cell to the goal, by breadth-first
   search outward from the goal; -1 where the goal cannot be reached.

Predict before you run: each cell is a wall with chance 0.2. Of 2,000 random
arenas, in how many can the agent not reach the goal at all? Fill in the
``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_grid.py
    uv run pytest tests/test_solutions.py -k grid -v
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from grid import MOVES, N, make_arena


def step(walls: np.ndarray, pos: tuple, action: int) -> tuple:
    """The next cell; a move into a wall or off the grid stays put."""
    r, c = pos[0] + MOVES[action][0], pos[1] + MOVES[action][1]
    # TODO(you): return (r, c) as two ints if it lies on the grid and is not a wall; else pos.
    ...


def bfs(walls: np.ndarray, goal: tuple) -> np.ndarray:
    """Steps from every cell to the goal; -1 where it cannot be reached."""
    dist = np.full((N, N), -1, dtype=np.int64)
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        # TODO(you): for each action, the neighbour step() reaches; if it moved and has no
        # distance yet, give it dist[cell] + 1 and queue it. (Moves are reversible here,
        # so walking out from the goal gives the distance to the goal.)
        ...
    return dist


def main() -> None:
    rng = np.random.default_rng(0)
    unreachable, far = 0, 0
    for _ in range(2000):
        walls, start, goal = make_arena(rng)
        d = bfs(walls, goal)[start]
        unreachable += d < 0
        far += d > 14
    print(f"of 2,000 arenas: goal unreachable in {unreachable}, more than 14 steps away in {far}")


if __name__ == "__main__":
    main()
