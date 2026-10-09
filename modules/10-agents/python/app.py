"""Gradio agent playground — chat with the local smolagents CodeAgent and watch
it think.

Left: a chat box. Right: the agent's step trace (the code it wrote, the tool
observations, the final answer) for the last turn. This makes the loop visible —
the whole point of the module.

    uv run python app.py            # launches on http://127.0.0.1:7860

The agent is 100% local (SmolLM2-1.7B on MPS); the first message is slow while
weights load. This is a teaching toy, not a product: a 1.7B model will sometimes
flail. That is honest and intentional.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

import model_local
from smol_way import build_agent

_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


def _format_trace(agent) -> str:
    lines = []
    for step in agent.memory.steps:
        num = getattr(step, "step_number", None)
        code = getattr(step, "code_action", None) or getattr(step, "model_output", None)
        obs = getattr(step, "observations", None)
        if code is None and obs is None:
            continue
        lines.append(f"### Step {num}")
        if code:
            lines.append("```python\n" + code.strip() + "\n```")
        if obs:
            lines.append("**Observation:** " + obs.strip()[:500])
    return "\n\n".join(lines) or "_(no steps recorded)_"


def respond(message, history):
    """Run the agent on ``message``; return (updated history, trace markdown)."""
    agent = _get_agent()
    try:
        answer = agent.run(message)
    except Exception as exc:  # noqa: BLE001
        answer = f"(agent error: {exc})"
    trace = _format_trace(agent)
    history = history + [(message, str(answer))]
    return history, "", trace


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Module 10 — Agent Playground") as demo:
        gr.Markdown(
            "# Agent Playground\n"
            "A **local** smolagents `CodeAgent` (SmolLM2-1.7B on MPS) with a "
            "calculator, a knowledge-base lookup, and a file-lister. Watch the "
            "step trace on the right. First message is slow (weights load)."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=420, label="Chat")
                msg = gr.Textbox(placeholder="e.g. Look up the speed of light, then multiply it by 2.",
                                 label="Your task", submit_btn=True)
                gr.Examples(
                    examples=[
                        "Use the calculator to compute (2 ** 10 + sqrt(144)) * 3 - 100.",
                        "Look up the speed of light in the knowledge base.",
                        "Look up 'seconds in an hour' in the knowledge base, then multiply it by 2.",
                        "List the files in the module's figures directory.",
                    ],
                    inputs=msg,
                )
            with gr.Column(scale=2):
                trace = gr.Markdown("_Run a task to see the agent's steps._",
                                    label="Agent step trace")
        msg.submit(respond, [msg, chat], [chat, msg, trace])
    return demo


def main():
    build_demo().launch()


if __name__ == "__main__":
    main()
