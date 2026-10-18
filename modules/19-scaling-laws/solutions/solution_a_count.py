"""Reference solution, exercise (a): count N by hand, and the compute C = 6 N D."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)

N_LAYER = 4
SIZES = (32, 48, 64, 96, 160, 256)
TOKENS = 4_001_792                      # 977 steps of 32 x 128 characters


def block_params(d: int) -> int:
    """One block: qkv (d -> 3d), attention out (d -> d), MLP (d -> 4d -> d), two LayerNorms; biases included."""
    qkv = d * 3 * d + 3 * d
    attn_out = d * d + d
    mlp = d * 4 * d + 4 * d + 4 * d * d + d
    norms = 2 * 2 * d                   # each LayerNorm has a weight and a bias of size d
    return qkv + attn_out + mlp + norms


def non_embedding_params(d: int, n_layer: int = N_LAYER) -> int:
    """N: every block plus the final LayerNorm. The token and position tables are left out;
    the output head is tied to the token table, so it adds nothing."""
    return n_layer * block_params(d) + 2 * d


def compute(n: int, tokens: int) -> float:
    """C = 6 N D: about 2 N FLOPs per token forward, 4 N backward."""
    return 6.0 * n * tokens


def main() -> None:
    for d in SIZES:
        n = non_embedding_params(d)
        print(f"d {d:>3}: N {n:>9,}  ~12 L d^2 = {12 * N_LAYER * d * d:>9,}  C at 4M tokens {compute(n, TOKENS):.2e} FLOPs")
    print(f"N grows {non_embedding_params(256) / non_embedding_params(32):.1f}x from d 32 to d 256")


if __name__ == "__main__":
    main()
