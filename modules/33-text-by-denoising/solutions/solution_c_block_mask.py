"""Solution (c): the attention pattern that trains block diffusion in one pass.

    uv run python ../solutions/solution_c_block_mask.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import torch


def block_mask(T: int, L: int) -> torch.Tensor:
    """(2T, 2T) bool over [noisy line ; clean line]; True = row may attend to column.
    noisy block b sees noisy block b and clean blocks before b; clean block b sees clean
    blocks up to b; clean never sees noisy."""
    b = torch.arange(T) // L
    same = b[:, None] == b[None, :]
    before = b[None, :] < b[:, None]
    upto = b[None, :] <= b[:, None]
    return torch.cat([torch.cat([same, before], 1), torch.cat([torch.zeros_like(same), upto], 1)], 0)


def main() -> None:
    print(block_mask(4, 2).int())
    for L in (1, 4, 16, 128):
        m = block_mask(128, L)
        print(f"L = {L:3d}: {int(m.sum()):6d} of {m.numel()} pairs may attend")


if __name__ == "__main__":
    main()
