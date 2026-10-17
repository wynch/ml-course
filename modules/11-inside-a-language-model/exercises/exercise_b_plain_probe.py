"""Exercise (b): the probe's residuals with plain PyTorch hooks, no nnsight.

``src/probe.py`` reads the residual stream through nnsight. Do the same with
``register_forward_hook``: one hook on ``model.model.embed_tokens`` and one on
each of the 32 decoder layers, each keeping the last token's vector. Then write
the plateau rule the README uses.

Predict before you run: is the residual norm at layer 32 closer to 4, 200 or
1,600? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_plain_probe.py
    uv run pytest tests/test_solutions.py -k probe -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import numpy as np
import torch

SENTENCES = [
    "a gorgeous, witty, seductive movie.",
    "the plot is nothing but boilerplate clichés from start to finish",
    "it's a charming and often affecting journey.",
    "unflinchingly bleak and desperate",
]


@torch.no_grad()
def last_token_residuals_plain(model, tokenizer, sentences: list[str]) -> np.ndarray:
    """(n, n_layers + 1, d_model): embeddings, then each block's output, last token.

    One sentence per forward pass, so no padding to think about.
    """
    layers = model.model.layers
    rows = []
    for s in sentences:
        ids = tokenizer(s, return_tensors="pt").input_ids.to(model.device)
        seen = []

        def grab(module, inputs, output):
            # TODO(you): a decoder layer may return a tensor or a tuple whose
            # first item is the hidden state. Append its last-token vector
            # (as float32, on the CPU) to `seen`.
            ...

        # TODO(you): register `grab` on embed_tokens and on every layer.
        handles = []
        try:
            model.model(ids)          # the base model: no 49,152-wide head needed
        finally:
            for h in handles:
                h.remove()
        rows.append(torch.stack(seen).numpy())
    return np.stack(rows)


def plateau_layer(acc: list[float], points: float = 1.0) -> int:
    """First layer whose accuracy is within ``points`` percentage points of the best."""
    # TODO(you): accuracies are fractions (0.883), points are percentage points.
    raise NotImplementedError


def main() -> None:
    from common import load_hf_model, load_tokenizer, read_results

    model, tok = load_hf_model(), load_tokenizer()
    H = last_token_residuals_plain(model, tok, SENTENCES)
    print("residuals:", H.shape, "norm at layer 0, 16, 32:",
          [round(float(np.linalg.norm(H[0, l])), 1) for l in (0, 16, 32)])
    acc = read_results("probe")["accuracy_trained"]
    print("plateau at 1 point:", plateau_layer(acc), "| at 3 points:", plateau_layer(acc, 3.0))


if __name__ == "__main__":
    main()
