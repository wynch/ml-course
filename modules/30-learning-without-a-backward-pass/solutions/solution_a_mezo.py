"""Solution (a): the two-point estimate, with the direction drawn from a seed.

    uv run python ../solutions/solution_a_mezo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import torch


def nudge(params: list[torch.Tensor], seed: int, scale: float) -> None:
    """Add scale · z to the weights in place, z drawn fresh from the seed, one tensor at a time."""
    g = torch.Generator().manual_seed(seed)
    for p in params:
        p.add_(torch.randn(p.shape, generator=g), alpha=scale)


@torch.no_grad()
def projected_grad(loss_fn, params: list[torch.Tensor], seed: int, eps: float = 1e-3) -> float:
    """(L(θ + εz) − L(θ − εz)) / 2ε, leaving the weights where they were."""
    nudge(params, seed, +eps)
    lp = loss_fn()
    nudge(params, seed, -2 * eps)
    lm = loss_fn()
    nudge(params, seed, +eps)
    return (lp - lm) / (2 * eps)


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
