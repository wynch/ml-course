"""Reference solution, exercise (c): BFS steps to the goal, then a ridge probe for them."""

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
        if grid[y, x] == GOAL:
            return float(d)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] and grid[ny, nx] not in BLOCKED:
                seen[ny, nx] = True
                q.append((nx, ny, d + 1))
    return float("inf")


def ridge_r2(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, lam: float = 1e-2) -> float:
    """Fit w = (Xb'Xb + lam I)^-1 Xb'y on tr (Xb = X with a column of ones); R^2 on te."""
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)
    A = Xb[tr].T @ Xb[tr] + lam * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb[tr].T @ y[tr])
    pred = Xb[te] @ w
    return float(1 - ((y[te] - pred) ** 2).sum() / (((y[te] - y[te].mean(axis=0)) ** 2).sum() + 1e-12))


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
