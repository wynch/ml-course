"""Solution (c): the cross-entropy method (CEM), with the true world as the model.

    uv run python ../solutions/solution_c_cem.py
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
        cand = (torch.randn(SAMPLES, H, ACT_IN, generator=gen) * std + mean).clamp(-1, 1)
        cand[0] = mean.clamp(-1, 1)
        elite = cand[torch.topk(world.cost(cand), ELITES, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
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
