"""BitLinear by hand: ternary weights, 8-bit activations, a straight-through gradient.

The rules are BitNet b1.58's (Ma et al., 2024, arXiv:2402.17764):

    weights      g = mean |W| over the matrix;  code = clamp(round(W / g), -1, 1);  W' = code * g
    activations  per token: s = 127 / max |x|;   x' = clamp(round(x * s), -128, 127) / s

The model keeps a float "latent" copy of every weight. The forward pass uses
W' and x'; the backward pass treats both roundings as if they were the
identity (the straight-through estimator, Bengio et al., 2013), so the
gradient lands on the latent weights unchanged.

`torch.round` sends halves to the even neighbour, so a latent at exactly
0.5 * g becomes code 0. That detail matters in section 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5
# 3^5 = 243 <= 256: five ternary codes fit in one byte, so 8 / 5 bits per weight.
TERNARY_BITS = 1.6


def absmean(W: torch.Tensor) -> torch.Tensor:
    return W.abs().mean().clamp_min(EPS)


def ternary_codes(W: torch.Tensor) -> torch.Tensor:
    """Codes in {-1, 0, 1}, same dtype as W; the arithmetic runs in W's dtype."""
    return (W / absmean(W)).round().clamp(-1, 1)


def ternary(W: torch.Tensor) -> torch.Tensor:
    """W' = code * mean|W|: what the forward pass multiplies with."""
    return ternary_codes(W) * absmean(W)


def act_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-token absmax rounding of activations to signed `bits`-bit integers, then back."""
    qmax = 2 ** (bits - 1) - 1
    s = qmax / x.abs().amax(dim=-1, keepdim=True).clamp_min(EPS)
    return (x * s).round().clamp(-qmax - 1, qmax) / s


def ste(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Forward: xq. Backward: the gradient of x."""
    return x + (xq - x).detach()


class BitLinear(nn.Linear):
    """nn.Linear whose forward uses ternary weights and 8-bit inputs; the bias stays float."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(ste(x, act_quant(x)), ste(self.weight, ternary(self.weight)), self.bias)


BLOCK_LINEARS = ("attn.qkv", "attn.proj", "mlp.fc", "mlp.proj")


def block_linears(model):
    """(name, module) for the four linear layers in every block of module 04's GPT."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in BLOCK_LINEARS) and name.startswith("blocks"):
            yield name, mod


def to_bitnet(model):
    """Swap every block linear for a BitLinear in place; the embedding, norms and head stay float."""
    for _, mod in list(block_linears(model)):
        mod.__class__ = BitLinear
    return model


# ---------------------------------------------------------------- storage
def file_bytes(model, weight_bits: float, group: int | None = None) -> dict:
    """Bytes to store the model when block-linear weights take `weight_bits` each.

    group=None: one 16-bit scale per matrix (ternary). group=g: affine groups of g
    weights along each row, each with a 16-bit scale and a 16-bit bias.
    Everything else (embeddings, positions, norms, biases) is stored in 16 bits.
    """
    lin, meta = 0, 0
    names = set()
    for name, mod in block_linears(model):
        n = mod.weight.numel()
        lin += n
        names.add(name + ".weight")
        meta += 2 if group is None else (n // group) * 4
    other = sum(p.numel() for n, p in model.named_parameters() if n not in names)
    total = lin * weight_bits / 8 + meta + other * 2
    return {"bytes": int(round(total)), "linear_weights": lin, "other_params": other,
            "linear_bytes": round(lin * weight_bits / 8 + meta), "other_bytes": other * 2}
