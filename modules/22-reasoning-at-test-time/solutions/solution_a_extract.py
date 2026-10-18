"""Solution (a): read the final answer out of a model's reply.

The rule: the number after the last "the answer is" (any case), else the last
number in the text, else None. Commas, "$" and a trailing period go; "18.00"
becomes "18".

    uv run python ../solutions/solution_a_extract.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

# A number: optional minus (not right after a digit, so "16-3" gives 16 and 3),
# digits with optional thousands commas, optional decimals.
NUM = r"(?<![\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def canon(s: str) -> str:
    v = float(s.replace(",", ""))
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else repr(round(v, 6))


def extract(text: str) -> str | None:
    hits = re.findall(r"answer is[:\s]*\**\s*\$?\s*(" + NUM + ")", text, flags=re.IGNORECASE)
    if hits:
        return canon(hits[-1])
    nums = re.findall(NUM, text)
    return canon(nums[-1]) if nums else None


CHECKS = [
    ("She makes 9 * 2 = $18.\nThe answer is 18.", "18"),
    ("The answer is $1,250.", "1250"),
    ("16-3-4 = 9 eggs are left.", "9"),
    ("**Answer:** $18", "18"),
    ("The answer is 5. Recheck: the answer is 6.", "6"),
    ("I cannot tell.", None),
]

if __name__ == "__main__":
    for text, want in CHECKS:
        got = extract(text)
        print(f"{'ok ' if got == want else 'BAD'} {got!r:8} want {want!r:8} {text!r}")
