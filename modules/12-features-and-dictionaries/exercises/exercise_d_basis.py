"""Exercise (d): change the nonlinearity, watch the untrained edge.

Section 5 found that an untrained model's neurons already beat random
directions, 1.8x at k = 8, because GELU acts on each coordinate apart. This
exercise keeps the untrained model's pre-activations u and swaps what is
applied to them before top-k: nothing, ReLU, GELU on 4u, tanh.

1. ``keep_topk``: keep the k largest entries by absolute value, in the neuron
   basis, or in hQ and rotated back.
2. ``output_kept``: the share of the layer's output, P h, that top-k keeps.

Predict before you run: for each of the five, is the edge at k = 8 below 1,
near 1, or above GELU's 1.8x? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_d_basis.py
    uv run pytest tests/test_solutions.py -k basis -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch
import torch.nn.functional as F

import basis
from sae import windows
from tiny_gpt import CORPUS, load_tiny_gpt, untrained_tiny_gpt

ACTIVATIONS = {
    "GELU (the model's)": F.gelu,
    "none (identity)": lambda u: u,
    "ReLU": F.relu,
    "GELU of 4u": lambda u: F.gelu(4 * u),
    "tanh": torch.tanh,
}


def keep_topk(h: torch.Tensor, k: int, Q: torch.Tensor | None = None) -> torch.Tensor:
    """Keep the k entries of h (tokens, 512) largest in absolute value, zero the rest.
    With Q, do it on hQ and rotate back."""
    # TODO(you): z = h, or h @ Q; take the indices of the k largest |z| per row
    # (Tensor.topk), copy those entries into zeros (gather, scatter); rotate back
    # with Q.T when Q is given.
    ...


def output_kept(h: torch.Tensor, P: torch.Tensor, k: int, Q: torch.Tensor | None = None) -> float:
    """1 - sum ||P(h_k - h)||^2 / sum ||P h||^2; P is proj.weight, (128, 512)."""
    # TODO(you): two lines; rows of h are tokens, so P h is h @ P.T.
    ...


def rate_k8(H, Ps, Qs) -> float:
    neu = [sum(output_kept(h, P, k) for h, P in zip(H, Ps)) / len(H) for k in basis.KS]
    rnd = [sum(output_kept(h, P, k, Qs[l]) for l, (h, P) in enumerate(zip(H, Ps))) / len(H) for k in basis.KS]
    return basis.exchange_rate(neu, rnd, 8, increasing=True)


def main() -> None:
    _, tok = load_tiny_gpt("cpu")
    model = untrained_tiny_gpt(1337, "cpu")
    text = CORPUS.read_text()
    x, _ = windows(torch.tensor(tok.encode(text[len(text) - int(len(text) * 0.1) :])))
    U = basis.pre_activations(model, x[: basis.CLEAN_WINDOWS])
    Ps = [b.mlp.proj.weight for b in model.blocks]
    Qs = [basis.rotation(l, 0) for l in range(len(U))]
    with torch.no_grad():
        for name, act in ACTIVATIONS.items():
            print(f"{name:<20} untrained edge at k = 8: {rate_k8([act(u) for u in U], Ps, Qs):.2f}x")


if __name__ == "__main__":
    main()
