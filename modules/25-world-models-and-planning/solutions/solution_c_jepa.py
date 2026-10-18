"""Solution (c): the JEPA loss and the EMA update; then what the latent keeps.

    uv run python ../solutions/solution_c_jepa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import torch

import grid
from models import JEPA, Probe, ema_momentum, r2, target_latents, tensors


def jepa_loss(model: JEPA, obs, action, next_obs) -> torch.Tensor:
    """Mean squared error between predictor(encoder(obs), action) and target(next_obs), no gradient
    into the target."""
    pred = model.predictor(model.encoder(obs), action)
    with torch.no_grad():
        tgt = model.target(next_obs)
    return ((pred - tgt) ** 2).mean()


@torch.no_grad()
def ema_update(target, online, mom: float) -> None:
    """target <- mom * target + (1 - mom) * online, parameter by parameter."""
    for a, b in zip(target.parameters(), online.parameters()):
        a.mul_(mom).add_(b.detach(), alpha=1 - mom)


def train(steps: int = 1500, seed: int = 0) -> JEPA:
    data = grid.collect(1000, 25, seed=1)
    T = tensors(data)
    torch.manual_seed(seed)
    model = JEPA()
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for s in range(1, steps + 1):
        i = torch.randint(len(T["obs"]), (256,), generator=g)
        loss = jepa_loss(model, T["obs"][i], T["action"][i], T["next_obs"][i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update(model.target, model.encoder, ema_momentum(s, steps))
    return model.eval()


def main() -> None:
    model = train()
    data, held = grid.collect(1000, 25, seed=1), grid.collect(200, 25, seed=2)
    Ztr, Zte = target_latents(model, data["obs"]), target_latents(model, held["obs"])
    for name, key in (("agent cell", "pos"), ("goal cell", "goal")):
        p = Probe.fit(Ztr, data[key])
        print(f"probe R2, {name}: {r2(p(Zte), held[key]):.3f}")


if __name__ == "__main__":
    main()
