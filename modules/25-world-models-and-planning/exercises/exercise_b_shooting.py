"""Exercise (b): random shooting, with the true world as the model.

Random shooting: draw 64 random sequences of 8 actions, imagine each in the
model, score each by the sum over its 8 imagined cells of the Manhattan
distance to the goal, and take the first action of the lowest score. Replan
at every step. Here the "model" is the world itself, so every imagined cell
is right: what is left is the planner.

Predict before you run: with a perfect model, in how many of 100 test arenas
does the planner reach the goal within 20 steps? All of them? Fill in the
``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_shooting.py
    uv run pytest tests/test_solutions.py -k shooting -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from grid import step, test_arenas


def imagine(walls, pos, actions: np.ndarray) -> np.ndarray:
    """The true world as a model: (shots, H) actions -> (shots, H, 2) cells."""
    out = np.zeros(actions.shape + (2,), dtype=np.int64)
    for s, seq in enumerate(actions):
        p = pos
        for k, a in enumerate(seq):
            p = step(walls, p, int(a))
            out[s, k] = p
    return out


def shoot(walls, pos, goal, rng: np.random.Generator, shots: int = 64, horizon: int = 8) -> int:
    """First action of the random sequence with the lowest summed Manhattan distance (the first on a tie)."""
    actions = rng.integers(4, size=(shots, horizon))
    cells = imagine(walls, pos, actions)
    # TODO(you): score = sum over the horizon of |row - goal row| + |col - goal col|, one per sequence;
    # return the first action of the sequence with the lowest score, as an int (np.argmin).
    ...


def run(arena: dict, rng: np.random.Generator, budget: int = 20) -> bool:
    walls, pos, goal = arena["walls"], arena["start"], arena["goal"]
    for _ in range(budget):
        pos = step(walls, pos, shoot(walls, pos, goal, rng))
        if pos == goal:
            return True
    return False


def main() -> None:
    arenas = test_arenas(100, seed=3)
    wins = [run(a, np.random.default_rng(1000 + i)) for i, a in enumerate(arenas)]
    print(f"a perfect model with random shooting reached the goal in {sum(wins)} of 100 test arenas")


if __name__ == "__main__":
    main()
