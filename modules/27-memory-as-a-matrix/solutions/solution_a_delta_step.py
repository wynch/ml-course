"""Exercise (a), solution: two ways to write into a matrix memory, and why one is a gradient step.

A state S (d_v x d_k) stores values under keys: read with S @ k.
Linear attention writes by adding:   S <- S + b v k^T
The delta rule first removes what S already returns for k:
                                     S <- S + b (v - S k) k^T
That second line is one step of gradient descent, rate b, on the loss
1/2 ||S k - v||^2. Here you write both, check the gradient claim with
autograd, and write one key twice.

Predict before you run: write (k1, v_old), then eight other pairs, then
(k1, v_new), all random unit vectors in 32 dimensions. Read k1. Which value
does each rule return: v_old, v_new, or a blend? Give the cosine you expect
with v_new for each.

    uv run python ../solutions/solution_a_delta_step.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch


def linear_write(S, k, v, b: float = 1.0):
    """One linear-attention write. S (d_v, d_k), k (d_k,), v (d_v,)."""
    return S + b * torch.outer(v, k)


def delta_write(S, k, v, b: float = 1.0):
    """One delta-rule write."""
    return S + b * torch.outer(v - S @ k, k)


def sgd_step(S, k, v, b: float):
    """One gradient step on 1/2 ||S k - v||^2, through autograd."""
    S = S.detach().clone().requires_grad_(True)
    loss = 0.5 * ((S @ k - v) ** 2).sum()
    (g,) = torch.autograd.grad(loss, S)
    return (S - b * g).detach()


def main() -> None:
    torch.manual_seed(0)
    d = 32
    unit = lambda x: x / x.norm(dim=-1, keepdim=True)
    S = torch.randn(d, d, dtype=torch.float64)
    k, v = unit(torch.randn(d, dtype=torch.float64)), unit(torch.randn(d, dtype=torch.float64))
    gap = (delta_write(S, k, v, 0.3) - sgd_step(S, k, v, 0.3)).abs().max().item()
    print(f"delta write vs one SGD step at rate 0.3: max difference {gap:.1e}")

    keys = unit(torch.randn(9, d, dtype=torch.float64))
    vals = unit(torch.randn(10, d, dtype=torch.float64))        # vals[0] = v_old, vals[9] = v_new
    order = [(keys[0], vals[0])] + [(keys[i], vals[i]) for i in range(1, 9)] + [(keys[0], vals[9])]
    for name, write in (("linear", linear_write), ("delta", delta_write)):
        S = torch.zeros(d, d, dtype=torch.float64)
        for kk, vv in order:
            S = write(S, kk, vv)
        r = S @ keys[0]
        cos = lambda a, b: (a @ b / a.norm() / b.norm()).item()
        print(f"{name:7s} read of k1: cosine {cos(r, vals[9]):+.3f} with v_new, {cos(r, vals[0]):+.3f} with v_old")


if __name__ == "__main__":
    main()
