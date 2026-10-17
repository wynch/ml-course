"""Exercise (a), solution: one GRU step by hand, checked against nn.GRU.

nn.GRU packs its weights as weight_ih (3H, in) and weight_hh (3H, H), in the
order reset, update, candidate (r, z, n), with two bias vectors.

Predict before you run: a GRU's update gate z sits near 1 for one channel.
What happens to that channel's h from one step to the next?

    uv run python ../solutions/solution_a_gru_cell.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch


def gru_cell(x, h, w_ih, w_hh, b_ih, b_hh):
    """x (B, in), h (B, H) -> new h (B, H), in nn.GRU's convention."""
    gi = x @ w_ih.T + b_ih                      # (B, 3H): input parts of r, z, n
    gh = h @ w_hh.T + b_hh                      # (B, 3H): hidden parts
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)                # how much of h the candidate may read
    z = torch.sigmoid(i_z + h_z)                # how much of h to keep
    n = torch.tanh(i_n + r * h_n)               # the candidate
    return (1 - z) * n + z * h


def run_gru(x, gru: torch.nn.GRU):
    """Run a one-layer nn.GRU's weights over x (B, T, in) with gru_cell. Returns (B, T, H)."""
    h = torch.zeros(x.shape[0], gru.hidden_size)
    outs = []
    for t in range(x.shape[1]):
        h = gru_cell(x[:, t], h, gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0)
        outs.append(h)
    return torch.stack(outs, 1)


def main() -> None:
    torch.manual_seed(0)
    gru = torch.nn.GRU(16, 32, batch_first=True)
    x = torch.randn(4, 50, 16)
    with torch.no_grad():
        ref, _ = gru(x)
        mine = run_gru(x, gru)
    print(f"max |difference| against nn.GRU over 50 steps: {(mine - ref).abs().max():.2e}")
    with torch.no_grad():
        h = torch.randn(1, 32)
        gru.bias_hh_l0[32:64] = 20.0            # update gate near 1 for every channel
        h2 = gru_cell(torch.randn(1, 16), h, gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0)
    print(f"with z near 1, max |h_t - h_(t-1)| = {(h2 - h).abs().max():.2e}: the channel holds its value")


if __name__ == "__main__":
    main()
