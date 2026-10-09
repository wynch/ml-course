"""The PyTorch reference tiny-GPT — the yardstick the Zig engine is measured
against.

The model definition below MIRRORS module 04's
`modules/04-attention-transformer/python/src/model.py` (the inference-relevant
parts of it). We keep our own copy, rather than importing across modules, so
this module is self-contained and its forward pass is spelled out right next to
the Zig one it validates.

Crucially, we do NOT load module 04's `.pt` checkpoint. We rebuild the model
straight from the committed `artifacts/tiny_gpt_weights.bin` blob, using the
manifest in `tiny_gpt_config.json`. That means the reference here and the Zig
engine read the *exact same bytes* — the parity test compares two readers of
one file, which is the honest thing to check.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention — same as module 04."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch's default F.gelu is the exact (erf) form; the Zig engine
        # matches it with an erf approximation good to ~1e-7.
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # tied LM head

    @torch.no_grad()
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab)


# --------------------------------------------------------------------------- #
# Load straight from the exported blob (no .pt checkpoint involved).
# --------------------------------------------------------------------------- #

def _read_blob(artifacts: Path):
    config = json.loads((artifacts / "tiny_gpt_config.json").read_text())
    blob = (artifacts / "tiny_gpt_weights.bin").read_bytes()
    tensors = {}
    for t in config["tensors"]:
        off, cnt = t["offset"], t["count"]
        arr = np.frombuffer(blob[off : off + cnt * 4], dtype="<f4").reshape(t["shape"])
        tensors[t["name"]] = torch.from_numpy(arr.copy())
    return config, tensors


def load_reference(artifacts: Path) -> tuple[GPT, dict]:
    """Build the GPT and populate its weights directly from the blob."""
    config, tensors = _read_blob(artifacts)
    cfg = GPTConfig(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        d_model=config["d_model"],
    )
    model = GPT(cfg)
    sd = model.state_dict()
    # map manifest tensor names onto the module's state_dict keys
    for name, t in tensors.items():
        assert name in sd, f"unexpected tensor {name}"
        assert tuple(sd[name].shape) == tuple(t.shape), f"shape mismatch {name}"
        sd[name] = t
    # head.weight is tied to wte.weight (not in the manifest) — set it too
    sd["head.weight"] = tensors["wte.weight"]
    model.load_state_dict(sd)
    model.eval()
    tok = json.loads((artifacts / "tokenizer_chars.json").read_text())
    stoi = {c: i for i, c in enumerate(tok["chars"])}
    return model, stoi


def encode(prompt: str, stoi: dict) -> torch.Tensor:
    ids = [stoi[c] for c in prompt if c in stoi]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
