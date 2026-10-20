"""Solution (c): keys per token or per channel, with and without a rotation, on made-up keys.

    uv run python ../solutions/solution_c_kv_layout.py
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
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    signs = torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(seed)) * 2 - 1
    return signs[:, None].float() * H / n**0.5


def fake_keys(tokens: int = 256, dim: int = 64, outliers=(7, 40), size: float = 8.0, seed: int = 0) -> torch.Tensor:
    """Gaussian keys with a few channels `size` times larger, as real keys have."""
    K = torch.randn(tokens, dim, generator=torch.Generator().manual_seed(seed))
    K[:, list(outliers)] *= size
    return K


def fq(K: torch.Tensor, bits: int, layout: str, R: torch.Tensor | None = None) -> torch.Tensor:
    """Round K (tokens, dim) in groups of 64: along each token's channels, or along each channel's tokens."""
    X = K @ R if R is not None else K
    if layout == "token":
        Y = quantize_affine(X, bits, 64)[0]
    else:
        Y = quantize_affine(X.T.contiguous(), bits, 64)[0].T
    return Y @ R.T if R is not None else Y


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
