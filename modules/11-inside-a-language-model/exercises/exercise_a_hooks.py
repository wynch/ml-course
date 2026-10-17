"""Exercise (a): a reading hook and an editing hook, on module 04's char-GPT.

1. ``record_neuron``: register a forward hook on ``blocks[block].mlp.fc``,
   keep GELU of its output for one neuron, run the model, remove the hook.
2. ``scale_neuron_logits``: register a forward *pre*-hook on
   ``blocks[block].mlp.proj`` that multiplies one neuron's input by k, and
   return the logits.

Predict before you run: with k = 0, do the letters right after a space change
more than the others? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_hooks.py
    uv run pytest tests/test_solutions.py -k hooks -v
"""

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
        # TODO(you): append GELU(output) for this neuron, shape (T,), to `seen`.
        ...

    # TODO(you): register `hook` as a forward hook on the block's mlp.fc.
    handle = ...
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
        # TODO(you): clone inputs[0], multiply the neuron's column by k,
        # and return the new inputs as a tuple.
        ...

    # TODO(you): register `pre_hook` as a forward pre-hook on the block's mlp.proj.
    handle = ...
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
