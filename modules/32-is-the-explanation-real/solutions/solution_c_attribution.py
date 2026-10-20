"""Solution (c): activation x gradient in the frozen model, and a one-neuron ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))
from attribution import Frozen  # noqa: E402


def attribute(model, x: torch.Tensor, target: int):
    fr = Frozen(model, x)
    keep = {}

    def leaf(h):
        keep[0] = h.detach().requires_grad_(True)
        return keep[0]

    def track(h):
        h.retain_grad()
        keep[1] = h
        return h

    model.blocks[0].mlp.edit, model.blocks[1].mlp.edit = leaf, track
    try:
        with fr:
            logit = model(x)[0, -1, target]
            logit.backward()
    finally:
        model.blocks[0].mlp.edit = model.blocks[1].mlp.edit = None
    att = torch.stack([(keep[i] * keep[i].grad)[0] for i in (0, 1)]).detach().numpy()
    return att, float(logit.detach())


@torch.no_grad()
def ablate(model, x: torch.Tensor, target: int, layer: int, pos: int, neuron: int) -> float:
    before = float(model(x)[0, -1, target])

    def zero(h):
        h = h.clone()
        h[0, pos, neuron] = 0
        return h

    model.blocks[layer].mlp.edit = zero
    try:
        after = float(model(x)[0, -1, target])
    finally:
        model.blocks[layer].mlp.edit = None
    return before - after


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
