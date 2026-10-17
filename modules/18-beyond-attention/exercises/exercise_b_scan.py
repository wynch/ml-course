"""Exercise (b): the selective recurrence as a loop, against its matrix form.

src/models.py trains SelectiveBlock with a (T, T) matrix: every output is a
weighted sum of every earlier input. Generation uses the recurrence: one state
S per head, updated once per token. Here you write the recurrence as a loop
over t and check that both give the same outputs.

Predict before you run: the block's state per head is a (P, N) matrix. With
the sizes below (8 heads, P = 35, N = 16), how many floats does the recurrence
carry from one step to the next, and does that number depend on T?

    uv run python ../exercises/exercise_b_scan.py
    uv run pytest tests/test_solutions.py -k scan -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch

from models import SelectiveBlock


def selective_loop(block: SelectiveBlock, xs: torch.Tensor, extra: torch.Tensor):
    """The recurrence, step by step. xs (B, T, E), extra (B, T, 2N + H). Returns y (B, T, E) and the last S."""
    Bsz, T, E = xs.shape
    Bm, Cm, dt, log_a = block._params(extra)       # (B,T,N), (B,T,N), (B,T,H), (B,T,H)
    x = xs.view(Bsz, T, block.H, block.P)
    S = torch.zeros(Bsz, block.H, block.P, block.N)
    ys = []
    for t in range(T):
        # TODO(you): a = exp(log_a[:, t]), shaped to broadcast over (B, H, P, N).
        # TODO(you): S = a * S + (dt_t * x_t) outer B_t, per head.
        # TODO(you): y_t = S C_t (sum over N), reshaped to (B, E); append it to ys.
        ...
    return torch.stack(ys, 1), S


def main() -> None:
    torch.manual_seed(0)
    block = SelectiveBlock(140)                    # E = 280, 8 heads of 35 channels, N = 16
    xs = torch.randn(2, 128, block.E)
    extra = torch.randn(2, 128, 2 * block.N + block.H)
    with torch.no_grad():
        ref = block.core(xs, extra)
        mine, S = selective_loop(block, xs, extra)
    print(f"max |loop - matrix form| over 128 steps: {(mine - ref).abs().max():.2e}")
    print(f"state carried between steps: {S[0].numel():,} floats per sequence, whatever T is")


if __name__ == "__main__":
    main()
