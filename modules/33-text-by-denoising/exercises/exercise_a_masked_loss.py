"""Exercise (a): the masking process and the masked diffusion loss.

At time t each character of a line becomes [MASK] with probability t. The
denoiser guesses every masked character; the loss is the cross-entropy on
those positions only, weighted by 1/t, per character of the line.

Predict before you run: an untrained model guesses each character with
probability about 1/65. What loss does it get at t = 0.1, 0.5 and 1.0? Does
the 1/t weight make small t cost more or less than large t? Fill in the
``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_masked_loss.py
    uv run pytest tests/test_solutions.py -k masked -v
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
    # TODO(you): one uniform draw per character; masked where it falls under that line's t;
    # x_t = mask_id where masked, x0 elsewhere.
    ...


def masked_loss(logits: torch.Tensor, x0: torch.Tensor, masked: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """logits (B, T, V), x0 (B, T), masked (B, T), t (B,). Cross-entropy on the masked positions only,
    each line's sum divided by its t and by T; mean over the batch."""
    # TODO(you): per-position cross-entropy (F.cross_entropy wants the class axis second, reduction="none"),
    # keep the masked positions, sum per line, divide by t and by T, mean over lines.
    ...


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
