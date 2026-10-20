"""Exercise (c): keys per token or per channel, with and without a rotation, on made-up keys.

Write ``hadamard`` (Sylvester's construction, scaled to be orthogonal, rows
times random signs) and ``fq``, which rounds a (tokens, 64) key matrix in
groups of 64 along tokens' channels or along channels' tokens, optionally
inside a rotation: K -> round(K R) R^T.

Predict before you run: two of 64 channels are 8 times larger than the rest.
At 2 bits, which layout errs less? Does the rotation help per token? Per
channel? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_c_kv_layout.py
    uv run pytest tests/test_solutions.py -k layout -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch

from compress import quantize_affine


def hadamard(n: int, seed: int = 0) -> torch.Tensor:
    """Sylvester's Hadamard matrix / sqrt(n), rows times random signs: orthogonal."""
    H = torch.ones(1, 1)
    # TODO(you): grow H to n x n by the block rule [[H, H], [H, -H]].
    signs = torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(seed)) * 2 - 1
    return signs[:, None].float() * H / n**0.5


def fake_keys(tokens: int = 256, dim: int = 64, outliers=(7, 40), size: float = 8.0, seed: int = 0) -> torch.Tensor:
    """Gaussian keys with a few channels `size` times larger, as real keys have."""
    K = torch.randn(tokens, dim, generator=torch.Generator().manual_seed(seed))
    K[:, list(outliers)] *= size
    return K


def fq(K: torch.Tensor, bits: int, layout: str, R: torch.Tensor | None = None) -> torch.Tensor:
    """Round K (tokens, dim) in groups of 64: along each token's channels, or along each channel's tokens."""
    # TODO(you): rotate if R is given; round with quantize_affine(X, bits, 64)[0], which works
    # along rows (transpose for "channel"); rotate back with R.T.
    ...


def rel_error(K: torch.Tensor, K_hat: torch.Tensor) -> float:
    return float((K_hat - K).norm() / K.norm())


def main() -> None:
    K = fake_keys()
    R = hadamard(64)
    print("max|k| per channel, top / median:",
          f"{float(K.abs().amax(0).max() / K.abs().amax(0).median()):.1f} before,",
          f"{float((K @ R).abs().amax(0).max() / (K @ R).abs().amax(0).median()):.1f} after the rotation")
    for bits in (4, 3, 2):
        row = []
        for layout in ("token", "channel"):
            for rot in (None, R):
                row.append(f"{layout}{'+rot' if rot is not None else ''} {rel_error(K, fq(K, bits, layout, rot)):.3f}")
        print(f"  {bits} bits: " + " | ".join(row))


if __name__ == "__main__":
    main()
