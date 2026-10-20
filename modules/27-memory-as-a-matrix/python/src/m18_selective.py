"""Module 18's selective recurrence in its matrix form, copied from
modules/18-beyond-attention/python/src/models.py (SelectiveBlock.core), with
the block's projections taken out so it takes its inputs directly.

    y_t = sum_{s<=t} (C_t . B_s) * exp(sum_{r=s+1..t} log a_r) * dt_s x_s

Section 2 checks that this module's "decay" rule gives the same outputs with
q = C, k = B, v = x, b = dt and log alpha = log a.
"""

from __future__ import annotations

import torch


def selective_matrix_form(xs, Bm, Cm, dt, log_a):
    """xs (B, T, H, P); Bm, Cm (B, T, N), shared by the heads; dt, log_a (B, T, H). Returns (B, T, H, P)."""
    Bsz, T, H, P = xs.shape
    L = torch.cumsum(log_a, dim=1).transpose(1, 2)                        # (B, H, T)
    seg = L[..., :, None] - L[..., None, :]                               # (B, H, t, s)
    causal = torch.ones(T, T, dtype=torch.bool, device=xs.device).tril()
    decay = torch.exp(seg.masked_fill(~causal, -torch.inf))
    M = (Cm @ Bm.transpose(1, 2))[:, None] * decay                        # (B, H, t, s)
    u = (xs * dt[..., None]).transpose(1, 2)                              # (B, H, s, P)
    return (M @ u).transpose(1, 2)
