"""Solution (c): the best-so-far curve and the budget the winner needed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: E402


def best_so_far(rows: list[dict]) -> list[tuple[int, float]]:
    out, top = [], 0.0
    for r in rows:
        if r["event"] != "full":
            continue
        top = max(top, r["acc"])
        out.append((r["calls"], top))
    return out


def budget_to_best(rows: list[dict]) -> int:
    curve = best_so_far(rows)
    final = curve[-1][1]
    return next(c for c, a in curve if a == final)


if __name__ == "__main__":
    for path in sorted(common.RUNS.glob("*.jsonl")):
        rows = common.read_jsonl(path)
        print(f"{path.stem:12s} best {best_so_far(rows)[-1][1]:.3f} "
              f"reached at {budget_to_best(rows)} of 2400")
