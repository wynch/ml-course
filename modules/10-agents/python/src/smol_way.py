"""The same three tasks, the smolagents way — a local CodeAgent.

Where the hand-rolled loop parsed ``Action: tool[arg]`` text, smolagents uses
**code as action**: the model writes a small Python snippet that *calls* the
tools, and a sandboxed interpreter runs it. One snippet can call several tools,
loop, and do arithmetic — often collapsing our multi-step ReAct dance into a
single step.

Everything stays local: ``TransformersModel`` runs SmolLM2-1.7B-Instruct on MPS,
and the tools are the very same functions from ``tools.py``, wrapped with
smolagents' ``@tool`` decorator.

    uv run python src/smol_way.py
"""

from __future__ import annotations

import json
from pathlib import Path

from smolagents import CodeAgent, TransformersModel, tool

import tools as _t
import model_local

MODULE_DIR = Path(__file__).resolve().parent.parent.parent


# --- the same three tools, wrapped for smolagents -------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Args:
        expression: A Python arithmetic expression, e.g. "2 ** 10 + sqrt(144)".
    """
    return _t.calculator(expression)


@tool
def kb_lookup(key: str) -> str:
    """Look up a fact in the local knowledge base by key (case-insensitive).

    Args:
        key: The fact to look up, e.g. "speed of light".
    """
    return _t.kb_lookup(key)


@tool
def list_module_files(subdir: str) -> str:
    """List files inside this course module's directory.

    Args:
        subdir: Subdirectory under the module root, or "" for the root.
    """
    return _t.list_module_files(subdir)


def build_model(model_id: str = model_local.DEFAULT_MODEL) -> TransformersModel:
    """A local TransformersModel (SmolLM2 on MPS), greedy for reproducibility."""
    return TransformersModel(
        model_id=model_id,
        device_map=model_local.get_device(),
        max_new_tokens=256,
        do_sample=False,
    )


def build_agent(model_id: str = model_local.DEFAULT_MODEL, max_steps: int = 3,
                verbosity_level: int = 0) -> CodeAgent:
    """Construct a local CodeAgent with our three tools."""
    model = build_model(model_id)
    return CodeAgent(
        tools=[calculator, kb_lookup, list_module_files],
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )


TASKS = [
    "Use the calculator tool to compute (2 ** 10 + sqrt(144)) * 3 - 100.",
    "Look up the speed of light in the knowledge base and report the value.",
    "Look up 'seconds in a day' in the knowledge base, then multiply that number by 3 using the calculator.",
    "List the files in the module's 'exercises' directory.",
]


def _steps_summary(agent: CodeAgent) -> list[dict]:
    """Pull a compact per-step record out of the agent's memory for figures."""
    out = []
    for step in agent.memory.steps:
        # ActionStep is the interesting one; skip the initial TaskStep/SystemPrompt.
        code = getattr(step, "code_action", None) or getattr(step, "model_output", None)
        obs = getattr(step, "observations", None)
        num = getattr(step, "step_number", None)
        if code is None and obs is None:
            continue
        out.append({
            "step": num,
            "code": (code or "").strip()[:400],
            "observation": (obs or "").strip()[:300] if obs else "",
        })
    return out


def main():
    agent = build_agent()
    transcripts = MODULE_DIR / "transcripts"
    transcripts.mkdir(exist_ok=True)

    logbook = []
    for task in TASKS:
        print(f"\n{'='*70}\nTASK: {task}\n{'='*70}", flush=True)
        try:
            answer = agent.run(task)
        except Exception as exc:  # noqa: BLE001
            answer = f"(agent error: {exc})"
        print(f"ANSWER: {answer}", flush=True)
        logbook.append({
            "task": task,
            "answer": str(answer),
            "steps": _steps_summary(agent),
        })

    (transcripts / "smol_traces.json").write_text(json.dumps(logbook, indent=2), encoding="utf-8")
    # A readable text log too.
    lines = []
    for entry in logbook:
        lines.append(f"TASK: {entry['task']}")
        for s in entry["steps"]:
            lines.append(f"  -- step {s['step']} --")
            if s["code"]:
                lines.append("  CODE:\n    " + s["code"].replace("\n", "\n    "))
            if s["observation"]:
                lines.append("  OBS: " + s["observation"])
        lines.append(f"ANSWER: {entry['answer']}")
        lines.append("=" * 70)
    (transcripts / "smol_trace.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {transcripts/'smol_trace.txt'}")


if __name__ == "__main__":
    main()
