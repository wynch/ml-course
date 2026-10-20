"""Solution (b): one unmasking step, by confidence or at random.

    uv run python ../solutions/solution_b_unmask.py
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
    base, rem = divmod(n_masked, steps)
    return [base + (1 if i < rem else 0) for i in range(steps)]


def step(logits: torch.Tensor, x: torch.Tensor, mask_id: int, k: int, rule: str) -> torch.Tensor:
    """logits (B, T, V), x (B, T) with some mask_id. Greedy guess for every position (argmax);
    reveal k masked positions per line: the k with the highest guessed probability ('confidence'),
    or the first k masked positions from left to right ('l2r'). Other positions keep x."""
    guess = logits.argmax(-1)
    masked = x == mask_id
    if rule == "confidence":
        score = F.softmax(logits, -1).gather(-1, guess[..., None]).squeeze(-1)
    else:
        score = -torch.arange(x.size(1), dtype=torch.float32).expand(x.shape).clone()
    score = torch.where(masked, score, torch.full_like(score, -torch.inf))
    keep = torch.zeros_like(masked)
    keep.scatter_(1, score.topk(k, dim=1).indices, True)
    return torch.where(keep, guess, x)


def main() -> None:
    print(reveal_counts(10, 4), reveal_counts(128, 1), reveal_counts(128, 128)[:3])
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 5) * 3
    x = torch.tensor([[1, 5, 5, 2, 5, 5]])
    for rule in ("confidence", "l2r"):
        print(rule, step(logits, x, 5, 2, rule).tolist())


if __name__ == "__main__":
    main()
