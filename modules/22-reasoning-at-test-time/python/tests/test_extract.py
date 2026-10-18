"""Known-answer test of the answer reader: 50 hand-written replies, all must pass.

Written before any generation ran. Each case is a reply and the answer the rule
must read from it, including the rule's known blind spot ("3/4 of 100 = 75"
reads 3, because "the answer is" comes first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gsm8k import extract, gold, same  # noqa: E402

CASES = [
    # (output text, expected answer)
    ("She sells 9 eggs and makes 9 * 2 = $18.\nThe answer is 18.", "18"),
    ("The answer is 72.", "72"),
    ("Total is 3 + 4 = 7. The answer is 7", "7"),
    ("He spends $1,250 in all. The answer is $1,250.", "1250"),
    ("The answer is 2.5.", "2.5"),
    ("The temperature drops to -5 degrees. The answer is -5.", "-5"),
    ("16-3-4 = 9 eggs are left.", "9"),
    ("So the total is 120 pages", "120"),
    ("I cannot tell from the question.", None),
    ("", None),
    ("First 12 * 2 = 24. Then 24 + 12 = 36. The answer is 36.\n", "36"),
    ("The answer is 10. Wait, recheck: 0.2 x 50 = 10. The answer is 10.", "10"),
    ("The answer is 5. Actually she needs 100 - 50 - 30 - 15 = 5. The answer is 6.", "6"),
    ("The final answer is 48.", "48"),
    ("THE ANSWER IS 16.", "16"),
    ("The answer is: 35", "35"),
    ("The answer is **42**.", "42"),
    ("Thus the answer is $ 30.", "30"),
    ("The answer is 1,000,000.", "1000000"),
    ("The answer is 18.00.", "18"),
    ("The answer is 0.25", "0.25"),
    ("The answer is 0.", "0"),
    ("She has 3 apples, 4 pears, and 5 plums.", "5"),
    ("Each costs $3.50, so 4 cost $14.", "14"),
    ("The answer is 624 pages a year.", "624"),
    ("He earns 15 per hour for 8 hours, which is 120. So the answer is 120 dollars.", "120"),
    ("50% of 80 is 40. The answer is 40%.", "40"),
    ("The answer is 7.5 hours.", "7.5"),
    ("The answer is 12345.", "12345"),
    ("The difference is 20 - 35 = -15. The answer is -15.", "-15"),
    ("We get \\boxed{64}.", "64"),
    ("The answer is 3/4 of 100 = 75.", "3"),
    ("There are 5 boxes with 12 each, 60 in all.", "60"),
    ("The answer is 9.\n\nQuestion: Tom has 4 cats. How many legs?", "9"),
    ("Step 1: 2 x 16 = 32\nStep 2: 2 x 8 = 16\nStep 3: 32 + 16 = 48\nThe answer is 48.", "48"),
    ("The answer is 100.5.", "100.5"),
    ("Price after discount: $80. The answer is $80.00", "80"),
    ("the answer is 21", "21"),
    ("No numbers here, the answer is unknown.", None),
    ("The answer is unknown, but it is at least 4.", "4"),
    ("In 2023 he was 40, so in 2025 he is 42.", "42"),
    ("The answer is 1,234.5.", "1234.5"),
    ("The answer is 7 and not 8.", "7"),
    ("Total: 3,000 + 2,500 = 5,500", "5500"),
    ("The answer is 0.333", "0.333"),
    ("So she needs 6 more. The answer is 6.\n", "6"),
    ("He drove 60 miles per hour for 2.5 hours = 150 miles.", "150"),
    ("The answer is -0.5.", "-0.5"),
    ("Answer is 13.", "13"),
    ("The answer is 8 eggs. The answer is 9 eggs.", "9"),
]

GOLD_CASES = [
    ("Janet sells 16 - 3 - 4 = 9 duck eggs.\n#### 18", "18"),
    ("x\n#### 1,000", "1000"),
    ("x\n#### -3", "-3"),
]



def test_fifty_cases():
    assert len(CASES) == 50


@pytest.mark.parametrize("text,want", CASES)
def test_extract(text, want):
    got = extract(text)
    assert (got is None and want is None) or same(got, want), (got, want)


@pytest.mark.parametrize("text,want", GOLD_CASES)
def test_gold(text, want):
    assert gold(text) == want
