"""Exercise (c): the cross-entropy method (CEM), with the true world as the model.

CEM searches over plans of 5 blocks of 5 two-number actions (50 numbers). It
keeps a Gaussian over plans, starting at mean 0 and spread 1. Each round it
draws 300 plans, scores them in the model, keeps the 30 cheapest and refits
the Gaussian to them. After 10 rounds the mean is the plan. Here the model is
the world itself and the cost the squared straight-line distance from the
last position to the goal, so what is left is the search and the cost.

Predict before you run: out of 20 tasks, how many goals does it reach when
the goal is in the same room? When it is in the other room? Fill in the
``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_c_cem.py
    uv run pytest tests/test_solutions.py -k cem -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import numpy as np
import torch

import plan
from plan import ELITES, H, ROUNDS, SAMPLES, TrueWorld
from lewm import ACT_IN


def cem(world, gen: torch.Generator) -> torch.Tensor:
    """LeWM's CEM: (H, 10) plan. Each round: draw SAMPLES sequences from N(mean, std), clipped to
    [-1, 1]; the first is the mean itself; keep the ELITES cheapest; refit mean and std to them."""
    mean, std = torch.zeros(H, ACT_IN), torch.ones(H, ACT_IN)
    for _ in range(ROUNDS):
        # TODO(you): draw SAMPLES plans: torch.randn(SAMPLES, H, ACT_IN, generator=gen) * std + mean, clipped to
        # [-1, 1]; set the first plan to the (clipped) mean; score them with world.cost(cand); keep the ELITES
        # cheapest (torch.topk with largest=False); refit mean and std to them (over the elite dimension).
        ...
    return mean.clamp(-1, 1)


def main() -> None:
    tasks = plan.tasks()
    world = TrueWorld()
    for split in ("same", "cross"):
        wins = 0
        for i, t in enumerate(tasks[split][:20]):
            gen = torch.Generator().manual_seed(10_000 + i)
            pos = np.array(t["start"], np.float32)
            for _ in range(plan.REPLANS):
                world.start(pos, t["goal"])
                block = cem(world, gen)[0].numpy().reshape(-1, 2)
                for a in block:
                    pos = plan.W.step(pos[None], a[None])[0]
                    if np.hypot(*(pos - np.array(t["goal"]))) < plan.W.REACH:
                        break
                if np.hypot(*(pos - np.array(t["goal"]))) < plan.W.REACH:
                    wins += 1
                    break
        print(f"{split:5s} room: CEM with a perfect model and a straight-line cost reached {wins} of 20 goals")


if __name__ == "__main__":
    main()
