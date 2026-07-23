"""Exercise B — break the agent, then fix it with the system prompt.

There is a task the hand-rolled agent gets STUCK on: it calls `kb_lookup` with a
key the KB doesn't have, the tool replies "Did you mean: earth radius km?", and
the small model keeps re-issuing almost-the-same key forever until it hits
max_steps. (See the shipped transcript `transcripts/handrolled_trace.txt`.)

Your job: WITHOUT changing any tool code, add one or two rules to the system
prompt so the agent recovers — reads the suggestion and retries with the exact
suggested key. Then run before/after and compare the traces.

Run:  uv run python ../exercises/exercise_b_break_fix.py
Solution: ../../solutions/solution_b_break_fix.py
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

# TODO(you): write a patched system prompt. Start from SYSTEM_PROMPT and append
# a rule telling the agent that when an Observation says "Did you mean: X", it
# must retry with EXACTLY X (verbatim, lowercase, no apostrophes). Keep it short.
FIXED_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n# TODO(you): add a recovery rule here.\n"


def main():
    print("=== BEFORE (default prompt) ===")
    before = run_react(LOOPING_TASK, max_steps=8)
    print(trace_to_text(before))

    print("\n=== AFTER (your fixed prompt) ===")
    after = run_react(LOOPING_TASK, max_steps=8, system_prompt=FIXED_SYSTEM_PROMPT)
    print(trace_to_text(after))


if __name__ == "__main__":
    main()
