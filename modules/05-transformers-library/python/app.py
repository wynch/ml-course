"""Gradio playground: chat with SmolLM2-360M-Instruct and watch the next-token
distribution live.

Left: a chat UI with temperature / top-k / top-p sliders. Right: a bar panel of
the model's probability distribution over the *very next token* given the
current conversation — the raw material every sampling knob acts on.

    uv run python app.py

A headless build+launch+close is exercised in tests/test_app.py, so no server
is ever left running by the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr
import pandas as pd
import torch
import torch.nn.functional as F

from common import load_model_and_tokenizer


def next_token_dist(model, tokenizer, prompt: str, temperature: float, k: int = 15) -> pd.DataFrame:
    """Top-k next-token probabilities (after temperature) as a tidy DataFrame."""
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float().cpu()
    probs = F.softmax(logits / max(temperature, 1e-2), dim=-1)
    vals, idx = probs.topk(k)
    tokens, ps = [], []
    for i, v in zip(idx.tolist(), vals.tolist()):
        tokens.append(tokenizer.decode([i]).replace("\n", "\\n").replace(" ", "␣") or "∅")
        ps.append(round(float(v), 4))
    return pd.DataFrame({"token": tokens, "prob": ps})


def _history_to_messages(history: list[dict], user_msg: str) -> list[dict]:
    messages = [{"role": t["role"], "content": t["content"]} for t in history]
    messages.append({"role": "user", "content": user_msg})
    return messages


def build_demo():
    """Return (demo, generate_fn) — generate_fn is unit-testable without a server."""
    model, tokenizer = load_model_and_tokenizer()

    def generate(user_msg, history, temperature, top_k, top_p):
        history = history or []
        messages = _history_to_messages(history, user_msg)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Distribution over the next token BEFORE we sample the reply.
        dist = next_token_dist(model, tokenizer, prompt, temperature)

        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=200,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-2),
                top_k=int(top_k),
                top_p=float(top_p),
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        new_history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        return new_history, dist, ""

    with gr.Blocks(title="SmolLM2 sampling playground") as demo:
        gr.Markdown(
            "# SmolLM2-360M sampling playground\n"
            "Chat on the left; the **next-token distribution** (after your last "
            "message, before the model answers) on the right. Move the sliders "
            "and watch the distribution sharpen or flatten."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=420, label="SmolLM2-360M-Instruct")
                msg = gr.Textbox(placeholder="Ask SmolLM2 something…", label="your message")
                with gr.Row():
                    temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="temperature")
                    top_k = gr.Slider(1, 100, value=50, step=1, label="top-k")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top-p")
                clear = gr.Button("clear")
            with gr.Column(scale=2):
                gr.Markdown("### Next-token distribution (top 15, at current temperature)")
                dist_plot = gr.BarPlot(
                    value=pd.DataFrame({"token": [], "prob": []}),
                    x="token", y="prob", height=440, sort=None, label=None,
                )

        msg.submit(
            generate,
            inputs=[msg, chatbot, temperature, top_k, top_p],
            outputs=[chatbot, dist_plot, msg],
        )
        clear.click(lambda: ([], pd.DataFrame({"token": [], "prob": []}), ""),
                    outputs=[chatbot, dist_plot, msg])

    return demo, generate


def main() -> None:
    demo, _ = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
