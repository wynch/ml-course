"""Exercise (b): linear warmup, then cosine decay.

The multiplier on the base learning rate rises from 0 to 1 over the first
``warmup`` steps, then follows half a cosine from 1 down to 0 at ``total``.
The test checks your function against transformers'
``get_cosine_schedule_with_warmup`` at every step of section 2's run.

Predict before you run: with 1,570 steps and 157 of warmup, what multiplier
does step 0 get? So what does the very first update change? And at which
step is the multiplier 0.5? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_schedule.py
    uv run pytest tests/test_solutions.py -k schedule -v
"""

from __future__ import annotations

import math  # noqa: F401  (you will need it)


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    """Multiplier on the base rate at ``step`` (0-based)."""
    # TODO(you): during warmup, step / warmup.
    # TODO(you): after it, progress = (step - warmup) / (total - warmup), and 0.5 (1 + cos(pi progress)).
    ...


def main() -> None:
    total, warmup = 1570, 157
    for s in (0, 1, 78, 157, 863, 1569):
        print(f"step {s:5d}: x {warmup_cosine(s, warmup, total):.4f}")


if __name__ == "__main__":
    main()
