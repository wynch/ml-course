"""A matrix state as a key -> value memory: four write rules, two ways to compute them.

The state S is a (d_v, d_k) matrix per head. Every rule reads the same way,
o_t = S_t q_t, and writes with one line:

    linear attention   S_t = S_{t-1} + b_t v_t k_t^T
    decay              S_t = a_t S_{t-1} + b_t v_t k_t^T
    delta rule         S_t = S_{t-1} + b_t (v_t - S_{t-1} k_t) k_t^T
    gated delta rule   S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_t^T

a_t in (0, 1] is a decay (we pass log a_t), b_t in (0, 1] a write strength.
"decay" with one scalar a_t per head is module 18's selective recurrence
(Mamba-2's form) in this module's notation: B = k, C = q, x = v, dt = b.

The delta rule's write is one step of gradient descent, rate b_t, on the loss
1/2 ||S k_t - v_t||^2: its gradient in S is (S k_t - v_t) k_t^T. So the state
first removes what it already returns for k_t, then writes v_t.

``recurrent`` runs the loop, one token at a time: what generation does.
``chunked`` computes the same outputs a chunk of C tokens at a time, with
matrix products inside each chunk; this is how DeltaNet trains (Yang et al.,
2024, arXiv 2406.06484). Inside a chunk, write S_t = g_t S_0 + sum_{i<=t}
(g_t / g_i) u_i k_i^T, with g_t the product of the decays so far. The delta
rule's u_t depend on the earlier u_i through a lower-triangular system,
(I + A) U = diag(b) (V - diag(g) K S_0^T), with A[t, i] = b_t (g_t / g_i) k_t . k_i
for i < t. One triangular solve gives every u_t at once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

RULES = {
    # name: (delta, gated)
    "linear": (False, False),
    "decay": (False, True),
    "delta": (True, False),
    "gated_delta": (True, True),
}


def recurrent(q, k, v, beta, log_alpha, delta: bool, S0=None):
    """The loop. q, k (B, H, T, d_k); v (B, H, T, d_v); beta, log_alpha (B, H, T).

    Returns o (B, H, T, d_v) and the last state S (B, H, d_v, d_k).
    """
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, H, dv, dk) if S0 is None else S0
    outs = []
    for t in range(T):
        a = torch.exp(log_alpha[:, :, t])[..., None, None]               # (B, H, 1, 1)
        b = beta[:, :, t][..., None]                                      # (B, H, 1)
        kt, vt = k[:, :, t], v[:, :, t]
        S = a * S
        if delta:
            vt = vt - (S @ kt[..., None])[..., 0]                         # v - a S k
        S = S + (b * vt)[..., None] * kt[..., None, :]
        outs.append((S @ q[:, :, t, :, None])[..., 0])
    return torch.stack(outs, 2), S


def chunked(q, k, v, beta, log_alpha, delta: bool, chunk: int = 64, S0=None):
    """The same outputs as ``recurrent``, a chunk at a time. T must divide by ``chunk``."""
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    assert T % chunk == 0, "pad T to a multiple of the chunk"
    S = q.new_zeros(B, H, dv, dk) if S0 is None else S0
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    strict = causal.clone().fill_diagonal_(False)
    outs = []
    for c0 in range(0, T, chunk):
        sl = slice(c0, c0 + chunk)
        Q, K, V, b = q[:, :, sl], k[:, :, sl], v[:, :, sl], beta[:, :, sl]
        g = torch.cumsum(log_alpha[:, :, sl], dim=-1)                     # log g_t, (B, H, C)
        seg = g[..., :, None] - g[..., None, :]                           # log (g_t / g_i)
        Gam = torch.exp(seg.masked_fill(~causal, -torch.inf))             # 0 above the diagonal
        if delta:
            A = (b[..., None] * (K @ K.transpose(-1, -2)) * Gam).masked_fill(~strict, 0.0)
            rhs = b[..., None] * (V - torch.exp(g)[..., None] * (K @ S.transpose(-1, -2)))
            U = unit_lower_inverse(A) @ rhs
        else:
            U = b[..., None] * V
        O = torch.exp(g)[..., None] * (Q @ S.transpose(-1, -2)) + ((Q @ K.transpose(-1, -2)) * Gam) @ U
        outs.append(O)
        last = g[..., -1:]                                                # log g_C
        S = torch.exp(last)[..., None] * S + (U * torch.exp(last - g)[..., None]).transpose(-1, -2) @ K
    return torch.cat(outs, 2), S


def unit_lower_inverse(A):
    """(I + A)^-1 for a strictly lower-triangular A (..., C, C), C a power of two.

    Block forward substitution, log2(C) rounds of batched matrix products. Start
    from blocks of size 1 (each inverse is 1); at each round, join neighbouring
    blocks with
        [[T1, 0], [A21, I + A22]]^-1 = [[T1, 0], [-T2 A21 T1, T2]]
    where T1, T2 are the inverses already found. Every product involves true
    sub-inverses, so nothing grows past the size of the answer.

    torch's triangular solve gives the same answer but runs slowly on Apple's
    GPU. A shorter product formula, (I - A)(I + A^2)(I + A^4)..., is exact in
    exact arithmetic but loses all precision in float32 when keys are alike,
    since the powers of A grow like binomial coefficients; the first run of
    section 3 used it (see the README).
    """
    C = A.shape[-1]
    assert C & (C - 1) == 0, "C must be a power of two"
    batch = A.shape[:-2]
    T = torch.ones(*batch, C, 1, 1, device=A.device, dtype=A.dtype)      # C blocks of size 1
    b = 1
    while b < C:
        n = C // (2 * b)
        # diagonal blocks of A of size 2b: (..., n, 2b, 2b)
        D = A.reshape(*batch, n, 2 * b, n, 2 * b).diagonal(dim1=-4, dim2=-2).movedim(-1, -3)
        T1, T2 = T[..., 0::2, :, :], T[..., 1::2, :, :]
        low = -T2 @ D[..., b:, :b] @ T1
        top = torch.cat([T1, torch.zeros_like(T1)], dim=-1)
        bot = torch.cat([low, T2], dim=-1)
        T = torch.cat([top, bot], dim=-2)
        b *= 2
    return T[..., 0, :, :]


def write_read(keys, values, rule: str, beta: float = 1.0, alpha: float = 1.0, passes: int = 1):
    """No training: write n (key, value) pairs into one state, then read every key back.

    keys (n, d_k), values (n, d_v). Returns what the state returns for each key, (n, d_v).
    """
    delta, gated = RULES[rule]
    n, dk = keys.shape
    S = keys.new_zeros(values.shape[1], dk)
    a = alpha if gated else 1.0
    for _ in range(passes):
        for k, v in zip(keys, values):
            S = a * S
            w = v - S @ k if delta else v
            S = S + beta * torch.outer(w, k)
    return keys @ S.T


def unit(x, dim=-1):
    return F.normalize(x, dim=dim)
