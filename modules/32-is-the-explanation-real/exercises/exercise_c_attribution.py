"""Exercise (c): attribution by hand, and the ablation that tests it.

``Frozen(model, x)`` (from src/attribution.py) fixes the attention patterns,
the LayerNorm scales and which 8 neurons each TopK keeps, at their values on
the prompt x. Inside it the target logit is a linear function of every
neuron's activation, as long as no kept neuron's ReLU input crosses zero.
Write:

1. attribute(model, x, target): run the frozen model with block 0's code h0
   made a leaf tensor (detach, then requires_grad) and block 1's code h1 kept
   with retain_grad(); backpropagate the target logit at the last position;
   return activation x gradient for both layers, shape (2, T, 2048), and the
   logit. Use each block's ``mlp.edit`` hook to swap in or keep h.
2. ablate(model, x, target, layer, pos, neuron): the drop in the target logit
   when that one neuron is set to zero, with nothing frozen.

Predict first: for the neuron with the largest |attribution| on prompt 0 of
the trained model, how close is the real drop to the attribution? And on the
random-weight copy?

    uv run python ../exercises/exercise_c_attribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))
from attribution import Frozen  # noqa: E402,F401


def attribute(model, x: torch.Tensor, target: int):
    """(attribution (2, T, N) numpy array, target logit float)."""
    fr = Frozen(model, x)          # build it before setting the edit hooks: it runs the model once
    # TODO(you): set model.blocks[0].mlp.edit and model.blocks[1].mlp.edit, run inside `with fr:`,
    #            backpropagate the logit, clear both hooks, return activation x gradient.
    raise NotImplementedError


@torch.no_grad()
def ablate(model, x: torch.Tensor, target: int, layer: int, pos: int, neuron: int) -> float:
    # TODO(you): an edit hook that zeroes h[0, pos, neuron] on a copy of h; return logit before minus after.
    raise NotImplementedError


def demo(which: str) -> None:
    import data
    from attribution import prompts
    from dicts import get_model

    _, va = data.tokens()
    sep = data.Vocab().encode("<|endoftext|>")[0]
    x = torch.from_numpy(prompts(va, sep)[0])[None]
    model = get_model(which, "cpu")
    with torch.no_grad():
        target = int(model(x)[0, -1].argmax())
    att, logit = attribute(model, x, target)
    l, p, n = np.unravel_index(np.argmax(np.abs(att)), att.shape)
    print(f"{which}: largest |attribution| is layer {l}, position {p}, neuron {n}: "
          f"{att[l, p, n]:+.3f}; real drop {ablate(model, x, target, l, p, n):+.3f}")


if __name__ == "__main__":
    demo("trained")
    demo("random")
