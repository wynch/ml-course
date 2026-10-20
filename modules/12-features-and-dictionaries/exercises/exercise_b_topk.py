"""Exercise (b): the three pieces of a TopK sparse autoencoder.

1. ``topk_encode``: ReLU of the encoder output, keep the k largest per row,
   zero the rest.
2. ``fve``: fraction of variance explained.
3. ``project_grad``: remove from each column's gradient its part along the
   column, so the step does not try to grow or shrink it (src/sae.py does
   this in ``train_sae``). After the step, src/sae.py also divides each column
   by its norm (``renorm``): the projection alone keeps the norm only to first
   order, and not at all once Adam rescales the step.

Predict before you run: what is the L0 of a TopK SAE with k = 4, before any
training? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_topk.py
    uv run pytest tests/test_solutions.py -k topk -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch
import torch.nn.functional as F


def topk_encode(x, W_enc, b_enc, b_pre, k: int) -> torch.Tensor:
    """z = TopK(ReLU(W_enc (x - b_pre) + b_enc)): keep the k largest, zero the rest.

    x (batch, d), W_enc (n, d), b_enc (n,), b_pre (d,). Returns z (batch, n).
    """
    pre = F.relu((x - b_pre) @ W_enc.T + b_enc)
    # TODO(you): keep the k largest entries of each row of `pre` (torch.topk),
    # put them back in place in a tensor of zeros (Tensor.scatter), return it.
    ...


def fve(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance explained: 1 - sum ||x - x_hat||^2 / sum ||x - mean(x)||^2."""
    # TODO(you): one line.
    ...


def project_grad(grad: torch.Tensor, W_dec: torch.Tensor) -> torch.Tensor:
    """Remove from each column's gradient its part along that (unit) column. W_dec, grad: (d, n)."""
    # TODO(you): per column j, subtract (grad_j . W_dec_j) * W_dec_j.
    ...


def main() -> None:
    torch.manual_seed(0)
    d, n, k = 16, 64, 4
    W_dec = F.normalize(torch.randn(d, n), dim=0)
    W_enc, b_enc, b_pre = W_dec.T.clone(), torch.zeros(n), torch.zeros(d)
    x = torch.randn(256, d)
    z = topk_encode(x, W_enc, b_enc, b_pre, k)
    print(f"L0 = {float((z > 0).sum(-1).float().mean()):.1f} (k = {k})")
    print(f"FVE of this untrained SAE: {fve(x, z @ W_dec.T + b_pre):.3f}")
    step = W_dec - 0.01 * project_grad(torch.randn(d, n), W_dec)
    print(f"column norms after a projected step: {float(step.norm(dim=0).min()):.5f} to {float(step.norm(dim=0).max()):.5f}")


if __name__ == "__main__":
    main()
