"""Exercise (b), solution: one chunk of the delta rule, all tokens at once.

Inside a chunk that starts from S = 0, the delta rule's state after t tokens is
S_t = sum_{i<=t} u_i k_i^T, where each u_t = b_t (v_t - S_{t-1} k_t) is the
error at step t, scaled. Each u_t depends on the earlier u_i:

    u_t + b_t sum_{i<t} (k_t . k_i) u_i = b_t v_t

In matrices, (I + A) U = diag(b) V, with A[t, i] = b_t k_t . k_i for i < t
and 0 elsewhere. A is strictly lower-triangular, so the system solves by
forward substitution: row t needs only the rows before it. Then every output
at once: O = tril(Q K^T) U (the diagonal included).

Predict before you run: the loop does C small steps, one per token, each
needing the last. How many steps does forward substitution need, and what
work can run in parallel inside each?

    uv run python ../solutions/solution_b_chunk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch

import memory


def forward_substitution(A, R):
    """Solve (I + A) X = R for a strictly lower-triangular A (..., C, C); R (..., C, d)."""
    C = A.shape[-1]
    rows = []
    for t in range(C):
        x = R[..., t, :]
        if t:
            x = x - (A[..., t, :t, None] * torch.stack(rows, -2)).sum(-2)
        rows.append(x)
    return torch.stack(rows, -2)


def delta_chunk(q, k, v, beta):
    """Outputs of the delta rule over one chunk from S = 0. q, k (..., C, d_k); v (..., C, d_v); beta (..., C)."""
    A = (beta[..., None] * (k @ k.transpose(-1, -2))).tril(-1)
    U = forward_substitution(A, beta[..., None] * v)
    return (q @ k.transpose(-1, -2)).tril() @ U


def main() -> None:
    torch.manual_seed(0)
    B, H, C, d = 2, 4, 64, 32
    q, k = memory.unit(torch.randn(B, H, C, d)), memory.unit(torch.randn(B, H, C, d))
    v, beta = torch.randn(B, H, C, d), torch.rand(B, H, C)
    mine = delta_chunk(q, k, v, beta)
    loop, _ = memory.recurrent(q, k, v, beta, torch.zeros_like(beta), delta=True)
    print(f"max |chunk - loop| over {C} tokens: {(mine - loop).abs().max():.1e}")
    A = (beta[..., None] * (k @ k.transpose(-1, -2))).tril(-1)
    gap = (forward_substitution(A, torch.eye(C).expand(B, H, C, C)) - memory.unit_lower_inverse(A)).abs().max()
    print(f"forward substitution vs the module's block form of (I + A)^-1: {gap:.1e}")


if __name__ == "__main__":
    main()
