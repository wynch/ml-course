"""A ReAct agent loop, hand-rolled in ~100 lines. No agent framework.

This is the whole idea of an agent, laid bare:

    prompt the model  ->  it writes a Thought and an Action
    parse the Action  ->  run the tool  ->  get an Observation
    inject the Observation back into the prompt  ->  repeat
    ... until the model writes "Final Answer:".

That's it. Everything smolagents, LangGraph, or a frontier "agent" product does
is a more robust version of this loop. We drive it with a *local* SmolLM2 and be
honest about where a 1.7B model stumbles (see the README).

Run directly to execute the three demo tasks and write a transcript + trace:

    uv run python src/handrolled.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tools import TOOL_REGISTRY
import model_local

# --- the ReAct prompt -----------------------------------------------------
_TOOL_LINES = "\n".join(f"- {desc}" for _, desc in TOOL_REGISTRY.values())

SYSTEM_PROMPT = f"""You are a careful reasoning agent that solves tasks using tools.

You have exactly these tools:
{_TOOL_LINES}

Work in a strict loop. Each turn you output ONE of:
  Thought: <your reasoning>
  Action: <tool_name>[<single argument>]
or, when you know the final result:
  Thought: <your reasoning>
  Final Answer: <the answer>

Rules:
- Use a tool for every calculation and every fact. Do not do arithmetic in your head.
- The tool call MUST be on its own line starting with "Action:". Never put a tool
  call inside a Thought.
- After each Action you will be shown an Observation. Use it.
- Put the whole argument inside the square brackets, e.g. Action: calculator[2 ** 8].
- Stop after writing an Action; do not write the Observation yourself.
- When you have enough information, write Final Answer and stop.

Follow the format EXACTLY. Here are two worked examples.

Example 1 (just report a looked-up fact — do NOT do extra arithmetic, even if it is a number):
Task: Look up the number of hours in a week in the knowledge base and report it.
Thought: I will look up hours in a week with the kb_lookup tool.
Action: kb_lookup[hours in a week]
Observation: 168
Thought: The task only asks me to report it, so I will not compute anything.
Final Answer: 168

Example 2 (chain a lookup into a calculation):
Task: Look up the golden ratio in the knowledge base, then multiply it by 2.
Thought: I will look up the golden ratio with the kb_lookup tool.
Action: kb_lookup[golden ratio]
Observation: 1.618033988749895
Thought: Now I multiply it by 2 using the calculator.
Action: calculator[1.618033988749895 * 2]
Observation: 3.23606797749979
Thought: I have the result.
Final Answer: 3.23606797749979
"""

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


@dataclass
class Step:
    kind: str  # "thought" | "action" | "observation" | "answer" | "error"
    content: str
    tool: str | None = None


@dataclass
class Trace:
    task: str
    steps: list[Step] = field(default_factory=list)
    answer: str | None = None
    ok: bool = False
    n_model_calls: int = 0  # how many times we prompted the model (a cost proxy)

    def add(self, kind, content, tool=None):
        self.steps.append(Step(kind, content, tool))


def run_react(task: str, model_id: str = model_local.DEFAULT_MODEL,
              max_steps: int = 6, verbose: bool = False,
              temperature: float = 0.0, system_prompt: str | None = None) -> Trace:
    """Run the hand-rolled ReAct loop on one task and return its Trace.

    ``temperature=0`` is greedy/reproducible; the eval lab passes ``>0`` to
    sample. ``system_prompt`` overrides the default (exercise B swaps it).
    """
    trace = Trace(task=task)
    n_steps = 0
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
    ]

    for _ in range(max_steps):
        n_steps += 1
        # 1. Ask the model for the next Thought + Action. Stop before it can
        #    hallucinate its own Observation.
        raw = model_local.generate(messages, model_id=model_id,
                                    max_new_tokens=200, stop=["Observation:"],
                                    temperature=temperature)
        if verbose:
            print("--- model ---\n" + raw + "\n")

        # 2. Record the Thought, if any.
        thought = raw.split("Action:")[0].split("Final Answer:")[0]
        thought = thought.replace("Thought:", "").strip()
        if thought:
            trace.add("thought", thought)

        # 3. Did the model finish?
        final = _FINAL_RE.search(raw)
        if final:
            trace.answer = final.group(1).strip()
            trace.ok = True
            trace.add("answer", trace.answer)
            break

        # 4. Parse the Action.
        m = _ACTION_RE.search(raw)
        if not m:
            trace.add("error", "no parseable Action or Final Answer")
            # Nudge the model back onto the format and try again.
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             "Please reply with either an Action: tool[arg] "
                             "or a Final Answer: line."})
            continue

        tool_name, arg = m.group(1), m.group(2).strip()
        trace.add("action", arg, tool=tool_name)

        # 5. Run the tool -> Observation.
        entry = TOOL_REGISTRY.get(tool_name)
        if entry is None:
            obs = f"Error: unknown tool {tool_name!r}. Tools: {', '.join(TOOL_REGISTRY)}"
        else:
            try:
                obs = entry[0](arg)
            except Exception as exc:  # noqa: BLE001
                obs = f"Error running {tool_name}: {exc}"
        trace.add("observation", obs, tool=tool_name)

        # 6. Inject the Observation and loop.
        messages.append({"role": "assistant", "content": raw.strip()})
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    trace.n_model_calls = n_steps
    return trace


# --- transcript rendering -------------------------------------------------
def trace_to_text(trace: Trace) -> str:
    lines = [f"TASK: {trace.task}", "=" * 60]
    label = {"thought": "Thought", "action": "Action", "observation": "Observation",
             "answer": "Final Answer", "error": "Error"}
    for s in trace.steps:
        if s.kind == "action":
            lines.append(f"Action: {s.tool}[{s.content}]")
        else:
            lines.append(f"{label[s.kind]}: {s.content}")
    lines.append("=" * 60)
    lines.append(f"RESULT: {'OK' if trace.ok else 'NO ANSWER'}  ->  {trace.answer!r}")
    return "\n".join(lines)


# Three demo tasks that a local 1.7B SmolLM2 handles reliably at temperature 0.
# They are phrased *explicitly* (name the tool, name the exact fact): coaxing a
# small model onto rails is a real part of building with one.
DEMO_TASKS = [
    "Use the calculator to evaluate this exact expression: sqrt(144) + 2 ** 10.",
    "Look up the capital of Japan in the knowledge base and report it.",
    "Look up 'seconds in a day' in the knowledge base, then use the calculator to multiply that number by 3.",
]

# One instructive failure the README dissects (multi-hop division trips the small model).
FAILURE_TASK = (
    "Look up the Earth's radius and the Moon's distance in the knowledge base, "
    "then compute how many Earth radii fit in the Earth-Moon distance."
)


def main():
    out_dir = Path(__file__).resolve().parent.parent.parent / "transcripts"
    out_dir.mkdir(exist_ok=True)
    all_text = []
    traces = []
    for task in DEMO_TASKS + [FAILURE_TASK]:
        print(f"\n### {task}")
        tr = run_react(task, verbose=False)
        traces.append(tr)
        rendered = trace_to_text(tr)
        print(rendered)
        all_text.append(rendered)

    (out_dir / "handrolled_trace.txt").write_text("\n\n\n".join(all_text), encoding="utf-8")
    # Dump the demo (non-failure) traces as JSON for the figure renderer.
    (out_dir / "handrolled_traces.json").write_text(
        json.dumps([
            {"task": t.task, "ok": t.ok, "answer": t.answer,
             "steps": [{"kind": s.kind, "content": s.content, "tool": s.tool} for s in t.steps]}
            for t in traces
        ], indent=2), encoding="utf-8")
    print(f"\nWrote transcript to {out_dir/'handrolled_trace.txt'}")


if __name__ == "__main__":
    main()
