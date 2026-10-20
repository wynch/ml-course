"""Solution (c): continual backprop's two small rules.

Each hidden unit keeps a running score of how much it contributes: its
average |activation| times the summed |weights| that carry it to the next
layer. Every step a unit owes a small share of a replacement; when a whole
replacement is owed, the lowest-scoring unit old enough to be judged gets new
input weights and zero output weights.

Predict first: a layer of 256 units, replacement rate 1e-4 per unit per step,
188 steps per task, 800 tasks. How many units are replaced in all? How many
times over is the layer renewed?

    uv run python ../solutions/solution_c_cbp.py
"""

from __future__ import annotations

import torch


def update_utility(util: torch.Tensor, h: torch.Tensor, w_out: torch.Tensor, decay: float) -> torch.Tensor:
    """util (units,), h (batch, units) activations, w_out (next_units, units). Returns the new running utility."""
    return decay * util + (1 - decay) * h.abs().mean(0) * w_out.abs().sum(0)


def pick_units(util: torch.Tensor, age: torch.Tensor, maturity: int, n: int) -> torch.Tensor:
    """Indices of the n lowest-utility units whose age is above `maturity` (fewer if not enough are mature)."""
    mature = (age > maturity).nonzero().squeeze(1)
    return mature[util[mature].argsort()[:n]]


def main() -> None:
    torch.manual_seed(0)
    util = torch.zeros(6)
    w_out = torch.tensor([[1.0, 0.1, 2.0, 0.0, 1.0, 0.5]] * 3)
    for _ in range(200):
        h = torch.relu(torch.randn(32, 6) + torch.tensor([1.0, 1.0, 0.0, 1.0, -3.0, 0.5]))
        util = update_utility(util, h, w_out, 0.99)
    print("utility:", [round(u, 3) for u in util.tolist()])
    age = torch.tensor([500, 500, 500, 500, 500, 50])
    print("replace 2 of them:", pick_units(util, age, maturity=100, n=2).tolist(),
          "(unit 5 is too young to be judged)")
    per_layer = 256 * 1e-4 * 188 * 800
    print(f"replacements per layer over 800 tasks: about {per_layer:.0f}, the layer renewed {per_layer / 256:.1f} times")


if __name__ == "__main__":
    main()
