"""Exercise (b): magnitude pruning of module 04's tiny char-GPT, on the CPU.

Write ``magnitude_mask``, then the script zeroes 0 to 90% of every block
matrix, by magnitude and at random, and prints the validation loss.

Predict before you run: module 04's model has 589,824 weights in its blocks.
At what fraction does magnitude pruning first cost more than 0.1 nats? And
does random pruning at 10% cost more or less than magnitude pruning at 50%?
Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_prune_tiny.py
    uv run pytest tests/test_solutions.py -k prune -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch
import torch.nn.functional as F

FRACTIONS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
VAL_WINDOWS = 200


def magnitude_mask(W: torch.Tensor, fraction: float) -> torch.Tensor:
    """Boolean mask of W's shape: False on the round(fraction * numel) weights of smallest |w|."""
    # TODO(you): k = round(fraction * numel); set False at the k positions of
    # smallest |w| (argsort of the flattened |W|), True elsewhere; reshape to W.
    ...


def block_weights(model):
    """The weight of every nn.Linear inside the transformer blocks (qkv, proj, fc, mlp proj)."""
    return [m.weight for b in model.blocks for m in b.modules() if isinstance(m, torch.nn.Linear)]


@torch.no_grad()
def val_loss(model, val: torch.Tensor) -> float:
    x = val[: VAL_WINDOWS * 128].view(VAL_WINDOWS, 128)
    y = val[1 : VAL_WINDOWS * 128 + 1].view(VAL_WINDOWS, 128)
    logits = model(x)[0]
    return float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)))


def main() -> None:
    from tiny_gpt import load_tiny_gpt, splits

    torch.manual_seed(0)
    model, tok = load_tiny_gpt("cpu")
    _, val = splits(tok)
    weights = block_weights(model)
    originals = [w.detach().clone() for w in weights]
    gen = torch.Generator().manual_seed(0)
    print(f"{len(weights)} matrices, {sum(w.numel() for w in weights):,} weights")
    for f in FRACTIONS:
        losses = []
        for kind in ("magnitude", "random"):
            for w, o in zip(weights, originals):
                if kind == "magnitude":
                    mask = magnitude_mask(o, f)
                else:
                    mask = torch.ones(o.numel(), dtype=torch.bool)
                    mask[torch.randperm(o.numel(), generator=gen)[: round(f * o.numel())]] = False
                w.data.copy_(o * mask.view_as(o))
            losses.append(val_loss(model, val))
        print(f"  {f:4.0%} zeroed: magnitude {losses[0]:.3f}, random {losses[1]:.3f} nats")


if __name__ == "__main__":
    main()
