"""Solution C — a conversation-summary memory for the hand-rolled loop.

`MemoryAgent` carries a one-line-per-turn summary and injects it into the next
run's system prompt, so "multiply that result by 10" resolves against the
previous answer. This is the seed of what real agents call short-term / episodic
memory — smolagents keeps a full step list; here we keep a lossy summary, which
is cheaper and keeps the context short for a small model.
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

    def _prompt(self) -> str:
        if not self.memory:
            return SYSTEM_PROMPT
        recap = "\n".join(f"- {line}" for line in self.memory)
        return (SYSTEM_PROMPT +
                "\nConversation so far (use it to resolve words like 'that'):\n" +
                recap + "\n")

    def run(self, task: str):
        trace = run_react(task, system_prompt=self._prompt(), max_steps=6)
        self.memory.append(f"Task: {task} -> Answer: {trace.answer}")
        return trace


def main():
    agent = MemoryAgent()
    t1 = agent.run("Use the calculator to compute 6 * 7.")
    print("A1:", t1.answer)
    t2 = agent.run("Use the calculator to multiply that result by 10.")
    print("A2:", t2.answer)
    print("\nMemory:")
    for line in agent.memory:
        print(" ", line)


if __name__ == "__main__":
    main()
