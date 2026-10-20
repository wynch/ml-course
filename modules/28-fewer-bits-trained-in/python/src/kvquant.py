"""The KV cache in fewer bits, by hand, in MLX.

`affine_fq` is module 23's affine rule (MLX's own: anchor the grid on the end
with the larger |x|, put zero on a level, round halves away from zero, store
scale and bias in bf16), written in MLX ops. It rounds along the last axis in
groups and returns the rebuilt values: "fake" quantisation, which measures
the damage and not the memory.

Two layouts for a cache tensor of shape (batch, kv_heads, tokens, head_dim):

    per token    each group is 64 channels of one token (head_dim is 64: one group per token per head)
    per channel  each group is 64 tokens of one channel

KIVI (Liu et al., 2024, arXiv:2402.02750) quantises keys per channel and
values per token. mlx-lm's QuantizedKVCache quantises both per token.

A rotation R (64 x 64, orthogonal) can be applied before rounding and undone
after: x -> fq(x R) R^T. For keys this equals rotating both queries and keys,
since (q R)(k R)^T = q k^T; QuaRot (Ashkboos et al., 2024, arXiv:2404.00456)
uses random Hadamard matrices for R.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

GROUP = 64


def round_half_away(x: mx.array) -> mx.array:
    return mx.sign(x) * mx.floor(mx.abs(x) + 0.5)


def affine_fq(x: mx.array, bits: int, group: int = GROUP) -> mx.array:
    """Quantise x along its last axis in groups of `group` and rebuild it; output in x's dtype."""
    shape, dtype = x.shape, x.dtype
    g = x.astype(mx.float32).reshape(*shape[:-1], shape[-1] // group, group)
    n_bins = 2**bits - 1
    w_max = mx.maximum(g.max(-1, keepdims=True), 0)
    w_min = g.min(-1, keepdims=True)
    low_end = mx.abs(w_min) > mx.abs(w_max)
    scale = mx.maximum((w_max - w_min) / n_bins, 1e-7)
    scale = mx.where(low_end, scale, -scale)
    edge = mx.where(low_end, w_min, w_max)
    q0 = round_half_away(edge / scale)
    scale = mx.where(q0 != 0, edge / q0, scale)
    bias = mx.where(q0 == 0, mx.zeros_like(edge), edge)
    q = mx.clip(round_half_away((g - bias) / scale), 0, n_bins)
    s16 = scale.astype(mx.bfloat16).astype(mx.float32)
    b16 = bias.astype(mx.bfloat16).astype(mx.float32)
    return (q * s16 + b16).reshape(shape).astype(dtype)


def hadamard(n: int, seed: int = 0) -> np.ndarray:
    """A random Hadamard matrix: Sylvester's H_n / sqrt(n), rows scaled by random signs. Orthogonal."""
    assert n & (n - 1) == 0, n
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=n)
    return (signs[:, None] * H / math.sqrt(n)).astype(np.float32)


def fq_cache_tensor(x: mx.array, bits: int, layout: str, R: mx.array | None = None) -> mx.array:
    """Round one cache tensor (batch, heads, tokens, head_dim) in the given layout, with an optional rotation."""
    dtype = x.dtype
    if R is not None:
        x = x.astype(mx.float32) @ R
    if layout == "token":
        y = affine_fq(x, bits)
    elif layout == "channel":
        y = affine_fq(x.swapaxes(-1, -2), bits).swapaxes(-1, -2)
    else:
        raise ValueError(layout)
    if R is not None:
        y = y.astype(mx.float32) @ R.T
    return y.astype(dtype)


class FakeQuantCache(KVCache):
    """A KV cache that stores keys and values after rounding them, by hand.

    Every value the model writes is rounded on the way in, so every query in a
    full-sequence pass attends to rounded keys and values, its own included.
    Per-channel keys need the token count to be a multiple of 64.
    """

    def __init__(self, bits: int, key_layout: str, value_layout: str = "token", R: mx.array | None = None):
        super().__init__()
        # Not "bits": mlx-lm sends any cache with a .bits attribute down its quantised-kernel path.
        self.n_bits, self.key_layout, self.value_layout, self.R = bits, key_layout, value_layout, R

    def update_and_fetch(self, keys, values):
        keys = fq_cache_tensor(keys, self.n_bits, self.key_layout, self.R)
        values = fq_cache_tensor(values, self.n_bits, self.value_layout, self.R)
        return super().update_and_fetch(keys, values)


class RecordingCache(KVCache):
    """A plain cache that also keeps a copy of the keys it was given (post-RoPE, as cached)."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def update_and_fetch(self, keys, values):
        self.seen.append(keys)
        return super().update_and_fetch(keys, values)
