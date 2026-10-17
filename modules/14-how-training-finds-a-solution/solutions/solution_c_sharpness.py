"""Reference solution, exercise (c): the top Hessian eigenvalue, without the Hessian."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch


def hvp(loss_fn, params, v):
    """H v for the Hessian H of loss_fn() with respect to params: differentiate (grad . v)."""
    grads = torch.autograd.grad(loss_fn(), params, create_graph=True)
    dot = sum((g * u).sum() for g, u in zip(grads, v))
    return torch.autograd.grad(dot, params)


def top_eigenvalue(loss_fn, params, iters=100, seed=0):
    """Power iteration: v <- H v / |H v|. Returns the Rayleigh quotient v . H v at the end."""
    g = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=g) for p in params]
    n = torch.sqrt(sum((u * u).sum() for u in v))
    v = [u / n for u in v]
    lam = 0.0
    for _ in range(iters):
        hv = hvp(loss_fn, params, v)
        lam = sum((h * u).sum() for h, u in zip(hv, v)).item()
        n = torch.sqrt(sum((h * h).sum() for h in hv))
        v = [h / n for h in hv]
    return lam


def main() -> None:
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(20, 20))
    A = Q @ torch.diag(torch.linspace(0.1, 7.0, 20)) @ Q.T
    x = torch.randn(20, requires_grad=True)
    lam = top_eigenvalue(lambda: 0.5 * x @ A @ x, [x])
    print(f"quadratic with eigenvalues 0.1 .. 7.0: power iteration gives {lam:.4f}")


if __name__ == "__main__":
    main()
