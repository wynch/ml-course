"""Exercise (a): the two-room step, with its wall and door.

The agent is a point in library units (0 to 224). An action in [-1, 1]^2
moves it by 5 units times the action. The border keeps its centre in
[21, 203]. A vertical wall at x = 112 splits two rooms; the agent may cross
only where y is in the door, [33.25, 64.75] once the slack is counted. A move
that would end inside the wall's reach (x past 100 from the left, or before
124 from the right) with y outside the door stops at 99.5 (or 124.5).

Predict before you run: of 100,000 random moves from random free points,
what share does the middle wall stop? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_world.py
    uv run pytest tests/test_solutions.py -k world -v
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
    # TODO(you): clip x and y of nxt to [LO, HI] (the border).
    x, y = ..., ...
    # TODO(you): the wall. in_door: is the new y within [DOOR_LO, DOOR_HI]? left: was the agent left of WALL_X
    # before the move? From the left, an x past WALL_L outside the door becomes WALL_L - 0.5; from the right,
    # an x before WALL_R outside the door becomes WALL_R + 0.5. Use np.where, one agent per row.
    ...
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
