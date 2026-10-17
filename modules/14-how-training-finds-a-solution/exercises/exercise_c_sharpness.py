"""Exercise (c): the top Hessian eigenvalue, without the Hessian.

Module 02's MLP has 203,530 parameters, so its Hessian has 4 x 10^10
entries: too big to build. A Hessian-vector product costs two backward
passes: take the gradient with ``create_graph=True``, dot it with v, and
differentiate that number again. Power iteration then needs only H v.

Predict before you run: the test quadratic has eigenvalues spread evenly from
0.1 to 7.0. What will power iteration return? Fill in the ``# TODO(you):``
lines, then:

    uv run python ../exercises/exercise_c_sharpness.py
    uv run pytest tests/test_solutions.py -k sharpness -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch


def hvp(loss_fn, params, v):
    """H v for the Hessian H of loss_fn() with respect to params: differentiate (grad . v)."""
    # TODO(you): grads = torch.autograd.grad(loss_fn(), params, create_graph=True)
    # TODO(you): dot = sum of (g * u).sum() over pairs; return torch.autograd.grad(dot, params)
    ...


def top_eigenvalue(loss_fn, params, iters=100, seed=0):
    """Power iteration: v <- H v / |H v|. Returns the Rayleigh quotient v . H v at the end."""
    g = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=g) for p in params]
    n = torch.sqrt(sum((u * u).sum() for u in v))
    v = [u / n for u in v]
    lam = 0.0
    for _ in range(iters):
        # TODO(you): hv = hvp(...); lam = v . hv; v = hv / |hv|
        ...
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
