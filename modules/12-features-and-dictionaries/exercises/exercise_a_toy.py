"""Exercise (a): the toy model of superposition, one model at a time.

1. ``toy_forward``: squeeze x into m dimensions with W, read it back with W^T,
   add b, apply ReLU.
2. ``toy_loss``: importance-weighted squared error.
3. ``represented``: count the columns of W with norm above 0.5.

Predict before you run: with 5 features in 2 dimensions, how many features
does the dense model (S = 0) keep? And the sparse one (S = 0.99)? Fill in the
``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_toy.py
    uv run pytest tests/test_solutions.py -k toy -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch

N, M = 5, 2                       # 5 features, 2 dimensions
STEPS = 5000


def toy_forward(W: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """One toy model: W (m, n), b (n,), x (batch, n). Returns ReLU(W^T W x + b), (batch, n)."""
    # TODO(you): h = x squeezed into m dimensions, then ReLU(h W + b).
    ...


def toy_loss(W, b, x, importance) -> torch.Tensor:
    """Importance-weighted squared error, summed over features, averaged over the batch."""
    # TODO(you): sum over features of importance * squared error, mean over the batch.
    ...


def represented(W: torch.Tensor, threshold: float = 0.5) -> int:
    """How many features have a column of norm above the threshold."""
    # TODO(you): count the columns W[:, i] whose norm is above the threshold.
    ...


def train_one(sparsity: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(M, N, generator=g) * 0.5)
    b = torch.nn.Parameter(torch.zeros(N))
    importance = torch.tensor([0.9**i for i in range(N)])
    opt = torch.optim.Adam([W, b], lr=1e-3)
    for _ in range(STEPS):
        x = torch.rand(1024, N, generator=g) * (torch.rand(1024, N, generator=g) >= sparsity)
        loss = toy_loss(W, b, x, importance)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach()


def main() -> None:
    for s in (0.0, 0.99):
        W = train_one(s)
        print(f"S = {s}: {represented(W)} of {N} features kept; column norms {W.norm(dim=0).numpy().round(2)}")


if __name__ == "__main__":
    main()
