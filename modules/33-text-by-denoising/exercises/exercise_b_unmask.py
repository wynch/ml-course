"""Exercise (b): one unmasking step, by confidence or by position.

A sampler starts from an all-[MASK] line and, at each step, guesses every
masked character at once but keeps only k of the guesses; the rest stay
masked for the next step. Which k to keep is the rule: LLaDA keeps the
guesses the model is most sure of.

Predict before you run: with the toy logits in main(), which two positions
does the confidence rule reveal? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_unmask.py
    uv run pytest tests/test_solutions.py -k unmask -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import torch
import torch.nn.functional as F


def reveal_counts(n_masked: int, steps: int) -> list[int]:
    """Split n_masked reveals over `steps` steps: equal shares, one more on each of the first
    n_masked % steps steps. reveal_counts(10, 4) == [3, 3, 2, 2]."""
    # TODO(you): divmod, then one extra reveal on each of the first `rem` steps.
    ...


def step(logits: torch.Tensor, x: torch.Tensor, mask_id: int, k: int, rule: str) -> torch.Tensor:
    """logits (B, T, V), x (B, T) with some mask_id. Greedy guess for every position (argmax);
    reveal k masked positions per line: the k with the highest guessed probability ('confidence'),
    or the first k masked positions from left to right ('l2r'). Other positions keep x."""
    guess = logits.argmax(-1)
    masked = x == mask_id
    # TODO(you): a score per position: the softmax probability of the guess ('confidence'), or minus the
    # position index ('l2r', so the leftmost scores highest); -inf where the position is not masked;
    # keep the top k per line (topk, then scatter True into a bool tensor); guess where kept, x elsewhere.
    ...


def main() -> None:
    print(reveal_counts(10, 4), reveal_counts(128, 1), reveal_counts(128, 128)[:3])
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 5) * 3
    x = torch.tensor([[1, 5, 5, 2, 5, 5]])
    for rule in ("confidence", "l2r"):
        print(rule, step(logits, x, 5, 2, rule).tolist())


if __name__ == "__main__":
    main()
