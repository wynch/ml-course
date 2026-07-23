"""Exercise C — give the hand-rolled loop a memory.

The ReAct loop in `handrolled.py` is amnesiac: each `run_react` call starts from
a blank slate. Add a tiny **conversation memory** so a second task can refer to
the first ("multiply THAT by 10"). Keep it simple: after each run, append a
one-line summary ("Task ... -> Answer ...") to a running memory string, and
inject that memory into the system prompt of the next run.

Run:  uv run python ../exercises/exercise_c_memory.py
Solution: ../../solutions/solution_c_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "python" / "src"
sys.path.insert(0, str(SRC))

from handrolled import run_react, SYSTEM_PROMPT  # noqa: E402


class MemoryAgent:
    def __init__(self):
        self.memory: list[str] = []

    def run(self, task: str):
        # TODO(you):
        #   1. Build a system prompt = SYSTEM_PROMPT + a "Conversation so far:"
        #      section listing self.memory (skip the section if memory is empty).
        #   2. Call run_react(task, system_prompt=...) and get the trace.
        #   3. Append f"Task: {task} -> Answer: {trace.answer}" to self.memory.
        #   4. Return the trace.
        raise NotImplementedError("implement MemoryAgent.run")


def main():
    agent = MemoryAgent()
    t1 = agent.run("Use the calculator to compute 6 * 7.")
    print("A1:", t1.answer)
    t2 = agent.run("Use the calculator to multiply that result by 10.")
    print("A2:", t2.answer)


if __name__ == "__main__":
    main()
