"""Reference solution, exercise (b): linear warmup, then cosine decay."""

from __future__ import annotations

import math


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    """Multiplier on the base rate at ``step`` (0-based)."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def main() -> None:
    total, warmup = 1570, 157
    for s in (0, 1, 78, 157, 863, 1569):
        print(f"step {s:5d}: x {warmup_cosine(s, warmup, total):.4f}")


if __name__ == "__main__":
    main()
