"""Solution B — fixing the KB retry loop from the system prompt.

The default prompt lets the small model loop: it re-issues "earth radius km"
(with the apostrophe still there) again and again. Adding one explicit recovery
rule — "when you see 'Did you mean: X', retry with EXACTLY X" — is usually enough
to break the loop, because it turns an open-ended reasoning step into a copy.

This is the whole teaching point of prompt-as-control-surface: you steer agent
behaviour with words, not code.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "python" / "src"
sys.path.insert(0, str(SRC))

from handrolled import run_react, trace_to_text, SYSTEM_PROMPT  # noqa: E402

LOOPING_TASK = (
    "Look up the Earth's radius and the Moon's distance in the knowledge base, "
    "then compute how many Earth radii fit in the Earth-Moon distance."
)

RECOVERY_RULE = """
Extra rules for recovering from a failed lookup:
- If an Observation contains "Did you mean:", your very next Action MUST call
  kb_lookup with EXACTLY the first suggested key, copied verbatim (all lowercase,
  no apostrophes). Do not add or keep words like "'s".
- Never repeat an Action that already failed. Change the argument.
- The keys you need here are exactly: earth radius km, moon distance km.
"""

FIXED_SYSTEM_PROMPT = SYSTEM_PROMPT + RECOVERY_RULE


def main():
    outdir = Path(__file__).resolve().parent.parent / "transcripts"
    outdir.mkdir(exist_ok=True)

    print("=== BEFORE (default prompt) ===")
    before = run_react(LOOPING_TASK, max_steps=8)
    print(trace_to_text(before))

    print("\n=== AFTER (fixed prompt) ===")
    after = run_react(LOOPING_TASK, max_steps=8, system_prompt=FIXED_SYSTEM_PROMPT)
    print(trace_to_text(after))

    (outdir / "exercise_b_before_after.txt").write_text(
        "=== BEFORE ===\n" + trace_to_text(before) +
        "\n\n=== AFTER ===\n" + trace_to_text(after), encoding="utf-8")


if __name__ == "__main__":
    main()
