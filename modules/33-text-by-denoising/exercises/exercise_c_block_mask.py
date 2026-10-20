"""Exercise (c): the attention pattern that trains block diffusion in one pass.

Block diffusion feeds the network the noisy line and the clean line side by
side, 2T positions, and lets attention decide who sees whom. A noisy block
must see itself and the clean blocks before it, never the clean copy of
itself (that would give the answer away) and nothing after it.

Predict before you run: how many of the 65,536 pairs may attend when L = 1,
and when L = 128? Which value of L turns this into plain masked diffusion?
Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_c_block_mask.py
    uv run pytest tests/test_solutions.py -k block -v
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
    # TODO(you): b = block index of each position; three (T, T) patterns from b: same block,
    # column's block before row's, column's block up to row's; place them in the four quarters.
    ...


def main() -> None:
    print(block_mask(4, 2).int())
    for L in (1, 4, 16, 128):
        m = block_mask(128, L)
        print(f"L = {L:3d}: {int(m.sum()):6d} of {m.numel()} pairs may attend")


if __name__ == "__main__":
    main()
