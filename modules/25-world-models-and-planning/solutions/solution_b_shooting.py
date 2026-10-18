"""Solution (b): random shooting, with the true world as the model.

    uv run python ../solutions/solution_b_shooting.py
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
    """Draw `shots` random sequences, score each by the summed Manhattan distance of its imagined
    cells to the goal, return the first action of the lowest score (the first one on a tie)."""
    actions = rng.integers(4, size=(shots, horizon))
    cells = imagine(walls, pos, actions)
    score = np.abs(cells - np.array(goal)).sum(-1).sum(1)
    return int(actions[int(np.argmin(score)), 0])


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
