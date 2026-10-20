"""Exercise (a): the two-point estimate, with the direction drawn from a seed.

MeZO never stores its random direction z. It draws z from a seed, nudges
every weight by +εz, measures the loss, nudges by −2εz, measures again,
nudges back by +εz, and draws the same z again from the same seed when it
updates. The scalar (L₊ − L₋) / 2ε times z is an estimate of the gradient.

Predict before you run: on a 20-weight quadratic, what cosine does one
direction give with the true gradient? And 100 directions, averaged? Fill in
the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_mezo.py
    uv run pytest tests/test_solutions.py -k mezo -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import torch


def nudge(params: list[torch.Tensor], seed: int, scale: float) -> None:
    """Add scale · z to the weights in place, z drawn fresh from the seed, one tensor at a time."""
    # TODO(you): one torch.Generator seeded with `seed`; for each tensor, add scale · randn of its shape
    # drawn from that generator, in place (p.add_(..., alpha=scale)).
    ...


@torch.no_grad()
def projected_grad(loss_fn, params: list[torch.Tensor], seed: int, eps: float = 1e-3) -> float:
    """(L(θ + εz) − L(θ − εz)) / 2ε, leaving the weights where they were."""
    # TODO(you): nudge by +eps, read loss_fn(), nudge by −2·eps, read it again, nudge back by +eps;
    # return (L₊ − L₋) / (2·eps). The weights must end where they started.
    ...


def estimate(loss_fn, params, n: int, eps: float = 1e-3) -> list[torch.Tensor]:
    """Average of g · z over n seeds: an estimate of the full gradient."""
    acc = [torch.zeros_like(p) for p in params]
    for s in range(n):
        g = projected_grad(loss_fn, params, s, eps)
        gen = torch.Generator().manual_seed(s)
        for a, p in zip(acc, params):
            a.add_(torch.randn(p.shape, generator=gen), alpha=g / n)
    return acc


def main() -> None:
    torch.manual_seed(0)
    A = torch.randn(20, 20)
    w = torch.randn(20)
    loss_fn = lambda: float(0.5 * ((A @ w) ** 2).sum())
    true = A.T @ (A @ w)
    for n in (1, 10, 100, 1000, 10000):
        est = estimate(loss_fn, [w], n)[0]
        cos = float(torch.nn.functional.cosine_similarity(est, true, 0))
        print(f"{n:6d} directions: cosine with the true gradient {cos:.3f}")


if __name__ == "__main__":
    main()
