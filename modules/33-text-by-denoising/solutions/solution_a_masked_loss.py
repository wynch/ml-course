"""Solution (a): the masking process and the masked diffusion loss.

    uv run python ../solutions/solution_a_masked_loss.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import torch
import torch.nn.functional as F

import data as D
from model import GPT, GPTConfig


def mask_line(x0: torch.Tensor, t: torch.Tensor, mask_id: int, gen=None):
    """x0 (B, T) ids; t (B,) masking rates. Each character becomes mask_id with probability t,
    independently. Returns (x_t, masked). Draw one uniform number per character with
    torch.rand(x0.shape, generator=gen), on the CPU."""
    u = torch.rand(x0.shape, generator=gen)
    masked = u < t[:, None]
    return torch.where(masked, torch.full_like(x0, mask_id), x0), masked


def masked_loss(logits: torch.Tensor, x0: torch.Tensor, masked: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """logits (B, T, V), x0 (B, T), masked (B, T), t (B,). Cross-entropy on the masked positions only,
    each line's sum divided by its t and by T; mean over the batch."""
    ce = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")
    return ((ce * masked).sum(1) / t / x0.size(1)).mean()


def main() -> None:
    tok, train, _ = D.load()
    torch.manual_seed(0)
    model = GPT(GPTConfig(tok.vocab_size, causal=False)).eval()
    gen = torch.Generator().manual_seed(0)
    x0 = D.batch(train, 32, gen)
    print(f"ln(65) = {math.log(65):.3f}: an untrained model's cross-entropy per guessed character")
    for t in (0.1, 0.5, 1.0):
        tt = torch.full((32,), t)
        xt, masked = mask_line(x0, tt, model.cfg.mask_id, gen)
        with torch.no_grad():
            loss = masked_loss(model(xt), x0, masked, tt)
        print(f"t = {t}: masked share {masked.float().mean():.3f}, loss {loss:.3f}")


if __name__ == "__main__":
    main()
