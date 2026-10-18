"""Solution (c): plain min/max group quantisation, and what the group size buys.

    uv run python ../solutions/solution_c_group_quant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch

GROUPS = (16, 32, 64, 128)


def quantize_minmax(W: torch.Tensor, bits: int, group: int) -> tuple[torch.Tensor, float]:
    """Affine quantisation per group of `group` weights along each row.

    b = min, s = (max - min) / (2^bits - 1), q = round((w - b) / s), w' = q * s + b.
    Returns (w', bits per weight), counting a 16-bit scale and a 16-bit bias per group.
    """
    out, d_in = W.shape
    Wg = W.reshape(out, d_in // group, group)
    lo = Wg.amin(-1, keepdim=True)
    hi = Wg.amax(-1, keepdim=True)
    s = ((hi - lo) / (2**bits - 1)).clamp_min(1e-12)
    q = ((Wg - lo) / s).round()
    return (q * s + lo).reshape(out, d_in), bits + 32 / group


def rel_error(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    return float((W_hat - W).norm() / W.norm())


def main() -> None:
    from compress import quantize_affine
    from tiny_gpt import load_tiny_gpt

    model, _ = load_tiny_gpt("cpu")
    W = model.blocks[1].mlp.fc.weight.detach()          # (512, 128)
    print(f"blocks[1].mlp.fc.weight {tuple(W.shape)}")
    for bits in (4, 3):
        for g in GROUPS:
            W_hat, bpw = quantize_minmax(W, bits, g)
            print(f"  {bits}-bit, group {g:3d}: {bpw:.2f} bits per weight, relative error {rel_error(W, W_hat):.4f}")
    mlx_rule = quantize_affine(W, 4, 64)[0]
    print(f"  MLX's rule, 4-bit group 64: relative error {rel_error(W, mlx_rule):.4f}")


if __name__ == "__main__":
    main()
