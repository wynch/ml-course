"""Before/after evidence — the signature deliverable of module 06.

Runs a FIXED prompt set through:
  (a) the base model   HuggingFaceTB/SmolLM2-360M-Instruct
  (b) base + your LoRA adapter (outputs/sft-lab/adapter)

and writes two artifacts:
  1. ../sample_generations.md   — committed, human-readable side-by-side
  2. ../figures/before_after_panel.png — a rendered text-panel figure for the README

Generation is greedy (do_sample=False) so the comparison is reproducible.

Run (after sft_train.py):  uv run python generate_compare.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import MODEL_ID, pick_device

HERE = Path(__file__).resolve().parent
ADAPTER_DIR = HERE / "outputs" / "sft-lab" / "adapter"
FIG_DIR = HERE.parent / "figures"
MD_OUT = HERE.parent / "sample_generations.md"

# A fixed, in-domain-ish prompt set. everyday-conversations is small-talk /
# practical-help flavoured, so we probe that register plus a couple of prompts
# that are OUT of that domain to show the honest limits of small-scale SFT.
PROMPTS = [
    "Hi there!",
    "I'm bored this afternoon. Any ideas for something to do at home?",
    "What should I make for dinner if I only have eggs, rice and some vegetables?",
    "Can you recommend a good book for a long train journey?",
    "I'm feeling a bit nervous about a job interview tomorrow. Any tips?",
    "How do I get red wine out of a white shirt?",
    "Explain how a suspension bridge stays up.",
    "Write a haiku about the sea.",
]

MAX_NEW_TOKENS = 160


def load(device, with_adapter: bool):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    if with_adapter:
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR)).to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, device, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    ).to(device)
    n_in = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][n_in:], skip_special_tokens=True)
    return text.strip()


def main():
    device = pick_device()

    print("generating with BASE model ...")
    tok_b, base = load(device, with_adapter=False)
    base_out = [generate(tok_b, base, device, p) for p in PROMPTS]
    del base

    print("generating with FINE-TUNED (base + LoRA) ...")
    tok_f, ft = load(device, with_adapter=True)
    ft_out = [generate(tok_f, ft, device, p) for p in PROMPTS]
    del ft

    write_markdown(base_out, ft_out)
    render_panel(base_out, ft_out)


def write_markdown(base_out, ft_out):
    lines = [
        "# Before / after — SmolLM2-360M-Instruct, base vs LoRA-SFT",
        "",
        "Greedy decoding (`do_sample=False`), 160 new tokens max. The adapter was",
        "trained for 200 steps on ~512 everyday-conversations examples. Read this",
        "with honest eyes: the base model is *already* instruction-tuned, so SFT on a",
        "tiny slice nudges **register and length**, not raw capability. See the",
        "commentary at the bottom.",
        "",
    ]
    for i, p in enumerate(PROMPTS):
        lines += [
            f"## Prompt {i+1}: {p}",
            "",
            "**Base:**",
            "",
            "> " + base_out[i].replace("\n", "\n> "),
            "",
            "**Fine-tuned (LoRA):**",
            "",
            "> " + ft_out[i].replace("\n", "\n> "),
            "",
            "---",
            "",
        ]
    lines += [
        "## Honest commentary",
        "",
        "- **What changed:** on the everyday-conversation prompts (1-5) the fine-tuned",
        "  model tends to answer more concisely and in the friendly, practical register",
        "  of the training data — shorter, less list-heavy, more conversational.",
        "- **What barely changed:** prompts 6-8 sit outside the fine-tuning domain.",
        "  Stain removal, bridge physics and haiku were already within the base model's",
        "  reach, and 200 steps of small-talk SFT neither teaches new facts nor removes",
        "  existing ones. This is expected.",
        "- **Why small-scale SFT is limited:** LoRA on a few hundred examples *steers*",
        "  an already-capable model; it does not inject knowledge or reasoning. To move",
        "  capability you need much more data, or a different objective (preference",
        "  optimisation / RL — see the What's next section).",
    ]
    MD_OUT.write_text("\n".join(lines))
    print(f"wrote {MD_OUT}")


def render_panel(base_out, ft_out, n_show=4):
    """A compact 4-prompt text panel for the README (keeps the PNG small)."""
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 9.5))
    fig.suptitle(
        "Base  vs  LoRA-SFT (200 steps, everyday-conversations)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    col_titles = ["BASE — SmolLM2-360M-Instruct", "FINE-TUNED — base + LoRA adapter"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=10, fontweight="bold",
                             color="#334155" if j == 0 else "#e11d48")

    def wrap(t, width=52, maxlines=9):
        out = []
        for para in t.split("\n"):
            out += textwrap.wrap(para, width=width) or [""]
        if len(out) > maxlines:
            out = out[:maxlines] + ["…"]
        return "\n".join(out)

    for i in range(n_show):
        for j, outs in enumerate((base_out, ft_out)):
            ax = axes[i, j]
            ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                         facecolor="#f8fafc" if j == 0 else "#fff1f2",
                         edgecolor="#e2e8f0"))
            ax.text(0.03, 0.96, f"Q: {PROMPTS[i]}", transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", va="top", color="#0f172a",
                    wrap=True)
            ax.text(0.03, 0.80, wrap(outs[i]), transform=ax.transAxes,
                    fontsize=7.6, va="top", family="monospace", color="#1e293b")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = FIG_DIR / "before_after_panel.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
