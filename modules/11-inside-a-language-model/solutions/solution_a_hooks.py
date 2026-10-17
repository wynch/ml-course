"""Reference solution, exercise (a): a reading hook and an editing hook."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch
import torch.nn.functional as F

from tiny_gpt import load_tiny_gpt

BLOCK, NEURON = 1, 139            # the space neuron found by src/hooks.py
TEXT = "PETRUCHIO:\nNow, by the world, it is a lusty wench;"


def record_neuron(model, ids: torch.Tensor, block: int, neuron: int) -> torch.Tensor:
    """Return the post-GELU value of one MLP neuron at every position, shape (T,)."""
    seen = []

    def hook(module, inputs, output):
        seen.append(F.gelu(output)[0, :, neuron].detach())

    handle = model.blocks[block].mlp.fc.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(ids[None])
    finally:
        handle.remove()
    return seen[0]


def scale_neuron_logits(model, ids: torch.Tensor, block: int, neuron: int, k: float) -> torch.Tensor:
    """Logits (T, vocab) with the neuron's post-GELU value multiplied by k.

    The hook sits on ``mlp.proj`` as a *pre*-hook, so it sees GELU's output
    directly and can scale it without inverting GELU.
    """
    def pre_hook(module, inputs):
        (h,) = inputs
        h = h.clone()
        h[..., neuron] *= k
        return (h,)

    handle = model.blocks[block].mlp.proj.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            logits, _ = model(ids[None])
    finally:
        handle.remove()
    return logits[0]


def main() -> None:
    model, tok = load_tiny_gpt()
    ids = torch.tensor(tok.encode(TEXT))
    acts = record_neuron(model, ids, BLOCK, NEURON)
    for c, a in zip(TEXT, acts):
        print(f"{c!r:>6} {a:+.2f} " + "#" * max(0, int(a * 10)))
    base = scale_neuron_logits(model, ids, BLOCK, NEURON, 1.0)
    for k in (0.0, 3.0):
        diff = (scale_neuron_logits(model, ids, BLOCK, NEURON, k) - base).abs().max()
        print(f"k = {k}: max logit change {diff:.3f}")


if __name__ == "__main__":
    main()
