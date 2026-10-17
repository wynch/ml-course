"""Exercise (a): SimCLR's NT-Xent loss.

Two augmented views of each of B images give 2B embeddings. Each view must
pick its partner out of the other 2B - 1 views: a cross-entropy over cosine
similarities divided by a temperature tau.

Predict before you run: if the encoder collapses and every embedding is the
same vector, what is the loss for B = 256? (Hint: what does a softmax over
2B - 1 equal numbers give?) Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_nt_xent.py
    uv run pytest tests/test_solutions.py -k nt_xent -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """z1, z2: (B, d), row i of each is a view of image i. Mean over the 2B views of
    -log softmax of the partner's cosine similarity / tau, against every other view."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    B = len(z1)
    # TODO(you): the (2B, 2B) similarity matrix divided by tau, with the diagonal set to -inf
    # (a view must not pick itself).
    sim = ...
    # TODO(you): the index of each view's partner: view i <-> view i + B.
    partner = ...
    return F.cross_entropy(sim, partner)


def main() -> None:
    torch.manual_seed(0)
    B, d, tau = 256, 128, 0.2
    a = torch.randn(B, d)
    print(f"two views identical:          {float(nt_xent(a, a, tau)):.4f}")
    print(f"two views unrelated:          {float(nt_xent(a, torch.randn(B, d), tau)):.4f}")
    c = torch.ones(B, d)
    print(f"collapsed (all rows equal):   {float(nt_xent(c, c, tau)):.4f}   log(2B - 1) = {math.log(2 * B - 1):.4f}")


if __name__ == "__main__":
    main()
