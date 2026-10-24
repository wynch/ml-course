"""Solution (b): the exact two-sided sign test on paired right/wrong vectors."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from common import read_results  # noqa: E402


def sign_test(a: list[int], b: list[int]) -> dict:
    wins = sum(1 for x, y in zip(a, b) if y == 1 and x == 0)
    losses = sum(1 for x, y in zip(a, b) if y == 0 and x == 1)
    n = wins + losses
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"wins": wins, "losses": losses, "p": p}


if __name__ == "__main__":
    base = read_results("baseline")
    print(sign_test(base["baseline"]["test_right"], base["by_hand"]["test_right"]))
