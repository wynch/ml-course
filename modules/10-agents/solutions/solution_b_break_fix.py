"""Solution B — fixing the KB retry loop from the system prompt, then the tool.

With the first, faulty `kb_lookup_naive`, the default prompt lets the small
model loop: it asks for "Earth's radius km", the tool misses (it normalises
"Earth's" to "earths") and suggests "earth radius km, moon distance km" as one
comma-joined string, and the model sends that whole string back, again and
again. Adding one explicit recovery rule — "when you see 'Did you mean:', retry
with EXACTLY the first suggested key" — turns an open-ended reasoning step into
a copy.

Then read the answer, not only the RESULT line: `ok` means only that the model
wrote a Final Answer. `check()` below compares it with 384400 / 6371 ≈ 60.3.

The third run keeps the default prompt but uses the fixed `kb_lookup`. Tool
output is part of the prompt; a tool built for the model that reads it can
remove the loop with no prompt rule at all.

This is the teaching point of prompt-as-control-surface, and its limit: you
steer agent behaviour with words, but the tools speak to the model too.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "python" / "src"
sys.path.insert(0, str(SRC))

from handrolled import run_react, trace_to_text, SYSTEM_PROMPT  # noqa: E402
from tools import NAIVE_TOOL_REGISTRY, TOOL_REGISTRY  # noqa: E402

LOOPING_TASK = (
    "Look up the Earth's radius and the Moon's distance in the knowledge base, "
    "then compute how many Earth radii fit in the Earth-Moon distance."
)
EXPECTED = 384400 / 6371  # ≈ 60.34 Earth radii

RECOVERY_RULE = """
Extra rules for recovering from a failed lookup:
- If an Observation contains "Did you mean:", your very next Action MUST call
  kb_lookup with EXACTLY the first suggested key, copied verbatim (all lowercase,
  no apostrophes). Do not add or keep words like "'s".
- Never repeat an Action that already failed. Change the argument.
"""

FIXED_SYSTEM_PROMPT = SYSTEM_PROMPT + RECOVERY_RULE


def check(trace) -> str:
    """Success check: a Final Answer within 1% of 384400 / 6371."""
    try:
        value = float(str(trace.answer).replace(",", "").split()[0])
    except (TypeError, ValueError, IndexError):
        return "WRONG (no number)"
    if abs(value - EXPECTED) / EXPECTED < 0.01:
        return "RIGHT"
    return f"WRONG (want ≈ {EXPECTED:.1f})"


def main():
    outdir = Path(__file__).resolve().parent.parent / "transcripts"
    outdir.mkdir(exist_ok=True)
    runs = [
        ("BEFORE (default prompt, naive tool)", None, NAIVE_TOOL_REGISTRY),
        ("AFTER (recovery rule, naive tool)", FIXED_SYSTEM_PROMPT, NAIVE_TOOL_REGISTRY),
        ("TOOL FIXED (default prompt, fixed tool)", None, TOOL_REGISTRY),
    ]
    parts = []
    for title, prompt, tools in runs:
        tr = run_react(LOOPING_TASK, max_steps=8, system_prompt=prompt, tools=tools)
        text = f"=== {title} ===\n" + trace_to_text(tr) + f"\nCHECK: {check(tr)}"
        print(text + "\n")
        parts.append(text)
    (outdir / "exercise_b_before_after.txt").write_text(
        "\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
