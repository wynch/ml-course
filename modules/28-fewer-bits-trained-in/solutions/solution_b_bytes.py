"""Solution (b): count a model's bytes, and find the shape that fills a budget.

    uv run python ../solutions/solution_b_bytes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

VOCAB, BLOCK = 317, 128


def model_bytes(d: int, n_layer: int, weight_bits: float, meta_bytes_per_weight: float, vocab: int = VOCAB) -> float:
    """Bytes of module 04's GPT at width d and depth n_layer.

    Block linears: 12 d^2 weights per layer (qkv 3d^2, proj d^2, fc 4d^2, mlp proj 4d^2), each
    `weight_bits` bits plus `meta_bytes_per_weight` bytes of scales and biases. Everything else
    in 16 bits: token and position embeddings (vocab + BLOCK) d, per layer two LayerNorms (4d) and
    four biases (3d + d + 4d + d = 9d), and the final LayerNorm (2d).
    """
    lin = 12 * d * d * n_layer
    other = (vocab + BLOCK) * d + 13 * d * n_layer + 2 * d
    return lin * weight_bits / 8 + lin * meta_bytes_per_weight + 2 * other


def width_for(budget: float, n_layer: int, weight_bits: float, meta: float, step: int = 32) -> int:
    """The multiple of `step` whose model is nearest the budget."""
    return min(range(step, 2048, step), key=lambda d: abs(model_bytes(d, n_layer, weight_bits, meta) - budget))


ARMS = {  # weight bits, bytes of scale and bias per weight
    "ternary": (1.6, 0.0),          # one 16-bit scale per matrix: negligible
    "4-bit, group 32": (4, 4 / 32),
    "2-bit, group 32": (2, 4 / 32),
    "16-bit": (16, 0.0),
}


def main() -> None:
    budget = 1.5e6
    print(f"budget {budget / 1e6} MB, 4 layers:")
    for arm, (bits, meta) in ARMS.items():
        d = width_for(budget, 4, bits, meta)
        b = model_bytes(d, 4, bits, meta)
        lin = 12 * d * d * 4
        print(f"  {arm:16s} d = {d:4d}: {b / 1e6:.3f} MB, {lin:,} block weights, "
              f"{100 * (b - lin * (bits / 8 + meta)) / b:.0f}% of the file outside the blocks")


if __name__ == "__main__":
    main()
