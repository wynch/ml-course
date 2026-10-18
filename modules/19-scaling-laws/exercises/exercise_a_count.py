"""Exercise (a): count the non-embedding parameters N by hand, and the compute C = 6 N D.

Module 04's block is: LayerNorm, attention (one d -> 3d matrix for q, k and v,
then a d -> d output matrix), LayerNorm, MLP (d -> 4d, GELU, 4d -> d). Every
Linear has a bias; every LayerNorm a weight and a bias. The model ends with one
more LayerNorm. N leaves out the token and position tables (the output head is
tied to the token table).

Predict before you run: d grows 8 times from 32 to 256. How many times does N grow?

    uv run python ../exercises/exercise_a_count.py
    uv run pytest tests/test_solutions.py -k count -v
"""

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
    # TODO(you): add up the four parts. Check: d = 32 gives 12,704.
    ...


def non_embedding_params(d: int, n_layer: int = N_LAYER) -> int:
    """N: every block plus the final LayerNorm."""
    # TODO(you): one line.
    ...


def compute(n: int, tokens: int) -> float:
    """C = 6 N D: about 2 N FLOPs per token forward, 4 N backward."""
    # TODO(you): one line.
    ...


def main() -> None:
    for d in SIZES:
        n = non_embedding_params(d)
        print(f"d {d:>3}: N {n:>9,}  ~12 L d^2 = {12 * N_LAYER * d * d:>9,}  C at 4M tokens {compute(n, TOKENS):.2e} FLOPs")
    print(f"N grows {non_embedding_params(256) / non_embedding_params(32):.1f}x from d 32 to d 256")


if __name__ == "__main__":
    main()
