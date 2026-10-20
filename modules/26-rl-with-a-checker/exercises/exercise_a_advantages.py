"""Exercise (a): three ways to turn a group's rewards into advantages.

Predict first: one problem, 8 replies, only the first is right
(rewards 1, 0, 0, 0, 0, 0, 0, 0). What advantage does the right reply get
under each rule? And the 7 wrong ones?

    uv run python ../exercises/exercise_a_advantages.py
"""

from __future__ import annotations

import torch


def adv_group(r: torch.Tensor) -> torch.Tensor:
    """GRPO: (r - group mean) / (group std + 1e-4); r is (problems, G); std with n - 1, torch's default."""
    # TODO(you): subtract each row's mean, divide by each row's std (torch's default, n - 1) + 1e-4.
    ...


def adv_no_std(r: torch.Tensor) -> torch.Tensor:
    """Dr. GRPO: r - group mean."""
    # TODO(you): subtract each row's mean.
    ...


def adv_loo(r: torch.Tensor) -> torch.Tensor:
    """RLOO: r_i - mean of the other G - 1 rewards."""
    # TODO(you): from each reward, subtract the mean of the other G - 1 rewards in its row.
    ...


def main() -> None:
    r = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    for name, f in [("group mean and std (GRPO)", adv_group), ("group mean (Dr. GRPO)", adv_no_std),
                    ("leave one out (RLOO)", adv_loo)]:
        print(name)
        for row, a in zip(r, f(r)):
            print("  rewards", row.int().tolist(), "-> advantages", [round(x, 2) for x in a.tolist()])
    ratio = adv_loo(r[:3]) / adv_no_std(r[:3])
    # Check your answer: the two should differ by one constant factor. Which?
    print("RLOO / Dr. GRPO, where both are non-zero:", sorted({round(x, 4) for x in ratio.flatten().tolist()}))


if __name__ == "__main__":
    main()
