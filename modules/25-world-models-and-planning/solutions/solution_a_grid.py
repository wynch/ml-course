"""Solution (a): the grid world's step and BFS.

    uv run python ../solutions/solution_a_grid.py
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
    if 0 <= r < N and 0 <= c < N and not walls[r, c]:
        return (int(r), int(c))
    return pos


def bfs(walls: np.ndarray, goal: tuple) -> np.ndarray:
    """Steps from every cell to the goal; -1 where it cannot be reached."""
    dist = np.full((N, N), -1, dtype=np.int64)
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for a in range(4):
            nxt = step(walls, cell, a)
            if nxt != cell and dist[nxt] < 0:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
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
