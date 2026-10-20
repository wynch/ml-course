"""LoRA by hand: a frozen linear layer plus a trainable low-rank change.

    y = W x + b + (alpha / r) * B (A x)

A is r x d_in, drawn small; B is d_out x r, zero at the start, so the wrapped
model starts out equal to the original. Only A and B train.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

TARGETS = ("attn.qkv", "attn.proj", "mlp.fc", "mlp.proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # as PEFT does

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.A.t() @ self.B.t())

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scale * self.B @ self.A


def add_lora(model: nn.Module, r: int, alpha: float | None = None) -> list[nn.Parameter]:
    """Freeze every weight of `model`, wrap each target linear layer, return the new weights."""
    alpha = float(r if alpha is None else alpha)
    for p in model.parameters():
        p.requires_grad_(False)
    new = []
    for name, module in list(model.named_modules()):
        for t in TARGETS:
            parent_name, _, child = t.partition(".")
            if name.endswith(parent_name) and isinstance(getattr(module, child, None), nn.Linear):
                wrapped = LoRALinear(getattr(module, child), r, alpha)
                setattr(module, child, wrapped)
                new += [wrapped.A, wrapped.B]
    return new
