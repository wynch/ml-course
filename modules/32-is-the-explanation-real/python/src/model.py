"""A two-layer transformer whose MLP neurons are the ground truth.

Each block's MLP is wide and TopK-sparse (after Korznikov et al. 2026,
arXiv 2610.19907, scaled down from 5 layers, d 256, 8,192 neurons, k 8):

    h   = TopK_k(ReLU(W_in · LN(x) + b_in))      N = 2,048 neurons, k = 8 kept per token
    out = W_out · h + b_out                      added to the residual stream

So every MLP output is b_out plus at most 8 columns of W_out, scaled. Those
columns are the **values** we score dictionaries against, and the rows of
W_in are the **keys**. Nothing about this is learned after the fact: the
model computes with these units by construction.

The dense baseline swaps the sparse MLP for a plain GELU MLP of 512 neurons.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 4096
    ctx: int = 128
    d: int = 128
    heads: int = 4
    layers: int = 2
    n_mlp: int = 2048
    k: int = 8                 # 0 means a dense GELU MLP
    def dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.h = c.heads
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.o = nn.Linear(c.d, c.d)
        self.register_buffer("mask", torch.tril(torch.ones(c.ctx, c.ctx, dtype=torch.bool)), persistent=False)
        self.frozen = None      # set to a pattern (B, H, T, T) to freeze attention for attribution
        self.last = None        # the last pattern computed

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.frozen is not None:
            p = self.frozen
        else:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
            s = s.masked_fill(~self.mask[:T, :T], float("-inf"))
            p = s.softmax(-1)
        self.last = p
        y = (p @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(y)


class SparseMLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.k = c.k
        self.W_in = nn.Linear(c.d, c.n_mlp)
        self.W_out = nn.Linear(c.n_mlp, c.d)
        self.edit = None        # optional function h -> h' (ablation, attribution)
        self.h = None           # the last hidden code, (B, T, N)
        self.gate = None        # set to a 0/1 mask (B, T, N) to freeze which neurons TopK keeps

    def code(self, x):
        pre = F.relu(self.W_in(x))
        if self.gate is not None:
            return pre * self.gate
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, i, v)

    def forward(self, x):
        h = self.code(x)
        if self.edit is not None:
            h = self.edit(h)
        self.h = h
        return self.W_out(h)


class DenseMLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.W_in = nn.Linear(c.d, c.n_mlp)
        self.W_out = nn.Linear(c.n_mlp, c.d)
        self.edit = None
        self.h = None

    def forward(self, x):
        h = F.gelu(self.W_in(x))
        if self.edit is not None:
            h = self.edit(h)
        self.h = h
        return self.W_out(h)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.d)
        self.mlp = SparseMLP(c) if c.k else DenseMLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab, c.d)
        self.wpe = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.ln_f = nn.LayerNorm(c.d)
        self.apply(self._init)
        for b in self.blocks:      # GPT-2's scaled init on the residual writes
            for w in (b.attn.o.weight, b.mlp.W_out.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x) @ self.wte.weight.T        # tied head


def random_copy(c: Config, seed: int = 1) -> GPT:
    """The same architecture with every weight at its initial random value (CPU draw, so it repeats)."""
    torch.manual_seed(seed)
    return GPT(c).eval()


def load(path, device="cpu") -> GPT:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = GPT(Config(**ck["config"]))
    m.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    return m.to(device).eval()


def save_half(model: GPT, path, extra: dict | None = None) -> None:
    """float16 weights: the committed copy, and the one every analysis loads."""
    torch.save({"config": model.c.dict(), "state": {k: v.detach().cpu().half() for k, v in model.state_dict().items()},
                **(extra or {})}, path)
