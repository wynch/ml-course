"""Reference solution, exercise (d): change the nonlinearity, watch the untrained edge.

Measured edges at k = 8 (one rotation, so GELU reads 1.81x, not section 5's 1.84x):
GELU 1.81x, none 1.00x, ReLU 2.04x, GELU of 4u 2.42x, tanh 0.79x.

- No nonlinearity: a rotation leaves u's distribution unchanged, so no basis is special.
- ReLU zeroes every negative entry at any scale, so its edge does not move with scale.
- GELU of 4u: over a wider input range GELU bends more, so the edge grows; here it
  even passes ReLU's (GELU of 10u gives about 2.1x, near ReLU again). GELU's edge depends
  on the scale of u, not only on the curve.
- tanh caps the large entries and keeps the small ones, so h is flatter than its
  rotation and neurons do worse than random directions.
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
    z = h if Q is None else h @ Q
    idx = z.abs().topk(k, dim=-1).indices
    zk = torch.zeros_like(z).scatter(-1, idx, z.gather(-1, idx))
    return zk if Q is None else zk @ Q.T


def output_kept(h: torch.Tensor, P: torch.Tensor, k: int, Q: torch.Tensor | None = None) -> float:
    """1 - sum ||P(h_k - h)||^2 / sum ||P h||^2; P is proj.weight, (128, 512)."""
    err = (keep_topk(h, k, Q) - h) @ P.T
    return 1 - float(err.pow(2).sum() / (h @ P.T).pow(2).sum())


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
