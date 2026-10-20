"""Solution (a): the two-room step, with its wall and door.

    uv run python ../solutions/solution_a_world.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from tworoom import DOOR_HI, DOOR_LO, HI, LO, SPEED, WALL_L, WALL_R, WALL_X, sample_free


def step(pos: np.ndarray, action: np.ndarray) -> np.ndarray:
    """(N, 2) positions and actions -> (N, 2) next positions, float32."""
    pos = np.asarray(pos, dtype=np.float32)
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    nxt = pos + a * np.float32(SPEED)
    x, y = np.clip(nxt[:, 0], LO, HI), np.clip(nxt[:, 1], LO, HI)
    in_door = (y >= DOOR_LO) & (y <= DOOR_HI)
    left = pos[:, 0] < WALL_X
    x = np.where(left & (x > WALL_L) & ~in_door, WALL_L - 0.5, x)
    x = np.where(~left & (x < WALL_R) & ~in_door, WALL_R + 0.5, x)
    return np.stack([x, y], 1).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(0)
    pos = sample_free(rng, 100_000)
    act = rng.uniform(-1, 1, (100_000, 2))
    nxt = step(pos, act)
    free = pos + np.clip(act, -1, 1) * SPEED
    stopped = (np.abs(nxt - free) > 1e-4).any(1)
    by_wall = stopped & np.isin(nxt[:, 0], [WALL_L - 0.5, WALL_R + 0.5])
    print(f"of 100,000 random moves from random free points, {stopped.mean():.1%} were stopped: "
          f"{by_wall.mean():.1%} by the middle wall, the rest by the border")


if __name__ == "__main__":
    main()
