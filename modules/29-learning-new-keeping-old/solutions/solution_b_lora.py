"""Solution (b): LoRA by hand.

Predict first: module 04's char-GPT has 623,104 weights here. Wrap its four
linear layers per block (qkv, attention out, MLP up, MLP down) with rank-4
LoRA. How many weights train? What fraction is that? Does the wrapped model
give the same output as the original before any step?

    uv run python ../solutions/solution_b_lora.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: E402,F401  (laptop guard before torch)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B (A x); W and b frozen, A small and random, B zero."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.A.t() @ self.B.t())

    def merged_weight(self) -> torch.Tensor:
        """The one matrix that does the same job: W + (alpha / r) B A."""
        return self.base.weight + self.scale * self.B @ self.A


def lora_count(d_model: int, n_layer: int, r: int) -> int:
    """Trainable weights of rank-r LoRA on qkv, proj, fc and mlp proj in every block."""
    shapes = [(d_model, 3 * d_model), (d_model, d_model), (d_model, 4 * d_model), (4 * d_model, d_model)]
    return n_layer * sum(r * (i + o) for i, o in shapes)


def main() -> None:
    torch.manual_seed(0)
    base = nn.Linear(128, 384)
    lo = LoRALinear(base, r=4, alpha=4)
    x = torch.randn(5, 128)
    print("same output before training:", torch.allclose(lo(x), base(x)))
    with torch.no_grad():
        lo.B.normal_()
    merged = x @ lo.merged_weight().t() + base.bias
    print("after a change, merged matrix gives the same output:", torch.allclose(lo(x), merged, atol=1e-5))
    print("rank of the change:", int(torch.linalg.matrix_rank(lo.merged_weight() - base.weight)))
    total = 623_104
    for r in (4, 32):
        n = lora_count(128, 3, r)
        print(f"rank {r}: {n:,} trainable weights, {100 * n / total:.1f}% of {total:,}")


if __name__ == "__main__":
    main()
