"""The compression rules, by hand: magnitude pruning and weight quantisation.

Every function takes one weight matrix W of shape (out, in) and returns a
new matrix of the same shape and dtype. Quantisers return the dequantised
weights (what the model computes with) plus what storing them would cost.

Quantisation works along the input dimension: a group is GROUP consecutive
weights of one row, and each group stores its own scale (and bias). A group
the size of the whole row is per-row ("per-channel") quantisation.
"""

from __future__ import annotations

import torch

PRUNED_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ----------------------------------------------------------------- pruning
def magnitude_mask(W: torch.Tensor, fraction: float) -> torch.Tensor:
    """Boolean mask, False on the round(fraction * numel) weights of smallest |w|."""
    k = round(fraction * W.numel())
    mask = torch.ones(W.numel(), dtype=torch.bool, device=W.device)
    if k:
        mask[W.abs().flatten().argsort()[:k]] = False
    return mask.view_as(W)


def random_mask(W: torch.Tensor, fraction: float, gen: torch.Generator) -> torch.Tensor:
    """Same count as magnitude_mask, positions drawn at random (a control)."""
    k = round(fraction * W.numel())
    mask = torch.ones(W.numel(), dtype=torch.bool)
    mask[torch.randperm(W.numel(), generator=gen)[:k]] = False
    return mask.view_as(W).to(W.device)


def block_linears(model):
    """(name, nn.Linear) for every attention and MLP matrix in the decoder blocks."""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in PRUNED_KINDS:
            yield name, mod


# ------------------------------------------------------------ quantisation
def quantize_symmetric(W: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, dict]:
    """Per-row symmetric: scale = max|w| / (2^(bits-1) - 1), q = round(w / scale)."""
    qmax = 2 ** (bits - 1) - 1
    Wf = W.float()
    scale = (Wf.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)
    scale = scale.to(torch.bfloat16).float()          # stored in bf16, like the checkpoint
    q = (Wf / scale).round().clamp(-qmax, qmax)
    deq = (q * scale).to(W.dtype)
    cost = {"weight_bits": bits, "groups": W.shape[0], "bytes": W.numel() * bits / 8 + W.shape[0] * 2}
    return deq, cost


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    """Metal's round(): halves go away from zero (torch.round sends them to even)."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def quantize_affine(W: torch.Tensor, bits: int = 4, group: int | None = 64) -> tuple[torch.Tensor, dict]:
    """Per-group affine, the rule of MLX's quantize kernel (mode "affine").

    Each group stores a scale s and a bias b; w is rebuilt as q * s + b with
    q in 0 .. 2^bits - 1. Plain min/max affine would set b = min and
    s = (max - min) / (2^bits - 1). MLX starts from that, then anchors the
    grid on whichever end has the larger |w| and nudges s so that zero falls
    exactly on a level. Two kernel details matter for a bit-exact match: its
    running max starts at 0 (so max >= 0), and it rounds halves away from
    zero. group=None means one group per row.
    """
    out, d_in = W.shape
    g = d_in if group is None else group
    assert d_in % g == 0, (d_in, g)
    n_bins = 2**bits - 1
    Wg = W.float().reshape(out, d_in // g, g)
    w_max = Wg.amax(-1, keepdim=True).clamp_min(0)
    w_min = Wg.amin(-1, keepdim=True)
    low_end = w_min.abs() > w_max.abs()
    scale = ((w_max - w_min) / n_bins).clamp_min(1e-7)
    scale = torch.where(low_end, scale, -scale)
    edge = torch.where(low_end, w_min, w_max)
    q0 = round_half_away(edge / scale)
    scale = torch.where(q0 != 0, edge / q0, scale)
    bias = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    q = round_half_away((Wg - bias) / scale).clamp(0, n_bins)
    s16, b16 = scale.to(torch.bfloat16).float(), bias.to(torch.bfloat16).float()   # stored in bf16
    deq = (q * s16 + b16).reshape(out, d_in).to(W.dtype)
    n_groups = out * (d_in // g)
    cost = {"weight_bits": bits, "groups": n_groups, "bytes": W.numel() * bits / 8 + n_groups * 4}
    return deq, cost


SCHEMES = {
    "int8, per row": lambda W: quantize_symmetric(W, 8),
    "4-bit, per row": lambda W: quantize_affine(W, 4, None),
    "4-bit, group 64": lambda W: quantize_affine(W, 4, 64),
}


def quantizable(model):
    """(name, weight Parameter) for every 2-D weight: the block matrices and the tied embedding."""
    for name, p in model.named_parameters():
        if p.dim() == 2:
            yield name, p
