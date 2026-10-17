"""Reference solution, exercise (c): patch one cell with plain hooks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch

CLEAN = "The Eiffel Tower is located in the city of"
CORRUPT = "The Colosseum is located in the city of"


def normalised_score(ld_patched: float, ld_clean: float, ld_corrupt: float) -> float:
    """0 = as the corrupt run, 1 = as the clean run."""
    return (ld_patched - ld_corrupt) / (ld_clean - ld_corrupt)


def site(model, row: int):
    """Row 0 = token embeddings, row k = the k-th block (``layers[k - 1]``)."""
    return model.model.embed_tokens if row == 0 else model.model.layers[row - 1]


@torch.no_grad()
def logit_diff(model, ids: torch.Tensor, a: int, b: int) -> float:
    logits = model(ids).logits[0, -1]
    return float(logits[a] - logits[b])


@torch.no_grad()
def patch_cell(model, tokenizer, row: int, position: int) -> dict:
    """Paste the clean residual at (row, position) into the corrupt run."""
    dev = model.device
    clean = tokenizer(CLEAN, return_tensors="pt").input_ids.to(dev)
    corrupt = tokenizer(CORRUPT, return_tensors="pt").input_ids.to(dev)
    a = tokenizer(" Paris", add_special_tokens=False).input_ids[0]
    b = tokenizer(" Rome", add_special_tokens=False).input_ids[0]

    saved = {}

    def save(module, inputs, output):
        saved["h"] = (output[0] if isinstance(output, tuple) else output).clone()

    def paste(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[0, position] = saved["h"][0, position]
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = site(model, row).register_forward_hook(save)
    try:
        ld_clean = logit_diff(model, clean, a, b)
    finally:
        handle.remove()
    ld_corrupt = logit_diff(model, corrupt, a, b)
    handle = site(model, row).register_forward_hook(paste)
    try:
        ld_patched = logit_diff(model, corrupt, a, b)
    finally:
        handle.remove()
    return {
        "ld_clean": ld_clean,
        "ld_corrupt": ld_corrupt,
        "ld_patched": ld_patched,
        "score": normalised_score(ld_patched, ld_clean, ld_corrupt),
    }


def main() -> None:
    from common import load_hf_model, load_tokenizer

    model, tok = load_hf_model(), load_tokenizer()
    for row, pos in ((3, 2), (3, 9), (27, 9), (32, 9), (20, 4)):
        r = patch_cell(model, tok, row, pos)
        print(f"row {row:>2}, position {pos}: score {r['score']:+.3f}")


if __name__ == "__main__":
    main()
