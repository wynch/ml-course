"""Module 04's char-level GPT, copied, with three changes for denoising.

1. ``causal`` is a switch. The autoregressive twin keeps the lower-triangular
   mask; the denoiser drops it, so every position sees every other one. That
   is the whole architectural change between an LLM and a masked diffusion
   LM (LLaDA's own code passes ``is_causal=False`` and nothing else).
2. The vocabulary gets one extra row, [MASK], with id ``vocab_size``. The
   head scores only the real characters: a denoiser never predicts [MASK].
3. ``forward`` takes an optional boolean attention mask (block diffusion
   needs its own pattern), explicit positions, and a KV cache.

Sizes are module 04's: 3 layers, 4 heads, width 128, context 128, dropout
0.1, tied embeddings; 0.62 M parameters. Attention calls
``F.scaled_dot_product_attention``; module 04 writes the same sum by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int               # real characters; [MASK] is id vocab_size
    block_size: int = 128
    n_layer: int = 3
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.1
    causal: bool = True

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    @property
    def mask_id(self) -> int:
        return self.vocab_size


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask=None, past=None):
        """x (B, T, C). attn_mask: bool (T, T_keys), True = may attend. past: (k, v) of earlier
        positions, each (B, H, T_past, d_head); they are prepended to this call's keys and values.
        Returns the output and this call's own (k, v), before the past is prepended."""
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.d_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (z.view(B, T, H, D).transpose(1, 2) for z in (q, k, v))
        own = (k, v)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        causal = self.cfg.causal and attn_mask is None and past is None
        drop = self.cfg.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), own


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, attn_mask=None, past=None):
        a, kv = self.attn(self.ln1(x), attn_mask, past)
        x = x + a
        return x + self.mlp(self.ln2(x)), kv


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size + 1, cfg.d_model)     # + [MASK]
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, attn_mask=None, pos=None, past=None, return_kv=False):
        """idx (B, T) ids, [MASK] allowed. pos (T,) positions, default 0..T-1. past: a list of
        per-layer (k, v) for earlier positions (the KV cache). Returns logits over the real
        characters (B, T, vocab_size), and with return_kv the per-layer (k, v) of this call."""
        B, T = idx.shape
        if pos is None:
            pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        kvs = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, attn_mask, None if past is None else past[i])
            kvs.append(kv)
        logits = self.ln_f(x) @ self.wte.weight[: self.cfg.vocab_size].T    # tied head, no [MASK] column
        return (logits, kvs) if return_kv else logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def extend_cache(cache, kvs):
    """Append one call's per-layer (k, v) to a cache (None = empty)."""
    if cache is None:
        return [(k, v) for k, v in kvs]
    return [(torch.cat([ck, k], 2), torch.cat([cv, v], 2)) for (ck, cv), (k, v) in zip(cache, kvs)]
