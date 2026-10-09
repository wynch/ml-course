"""Chat-template dissection: what actually gets fed to the model.

An instruct model was fine-tuned on a *specific* string format wrapping each
turn with special tokens (``<|im_start|>``, ``<|im_end|>``). Get that format
wrong and the model degrades. ``apply_chat_template`` builds the exact string;
here we render it and highlight every token boundary, colouring the special
tokens that mark system / user / assistant turns.

Run me:
    uv run python src/chat_template.py           # print the token walk
    uv run python src/chat_template.py --figure  # draw the highlighted boundaries
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_tokenizer

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"

MESSAGES = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def dissect(tokenizer):
    text = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    ids = enc["input_ids"]
    special_ids = set(tokenizer.all_special_ids)
    # Some special tokens (like <|im_start|>) live in added_tokens, not the
    # base special set — include everything the tokenizer flags as "added".
    added = {v for v in tokenizer.get_added_vocab().values()}
    pieces = []
    for tid in ids:
        piece = tokenizer.decode([tid])
        is_special = tid in special_ids or tid in added
        pieces.append({"id": tid, "text": piece, "special": is_special})
    return text, pieces


def fig_chat_template(tokenizer, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    _, pieces = dissect(tokenizer)

    fig, ax = plt.subplots(figsize=(11, 2.7))
    ax.axis("off")
    ax.set_ylim(0, 1)
    x, y = 0.01, 0.86
    line_h = 0.17
    ax.text(
        0.5, 0.99,
        "Chat template, tokenized — each box is one token; orange = special / structural",
        ha="center", va="top", fontsize=11, fontweight="bold", transform=ax.transAxes,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for pc in pieces:
        disp = pc["text"].replace("\n", "\\n").replace(" ", "␣") or "∅"
        color = "#DD8452" if pc["special"] else "#4C72B0"
        txt = ax.text(
            x + 0.010, y, disp, fontsize=10, fontfamily="monospace",
            color="white", va="center", transform=ax.transAxes, zorder=3,
        )
        bb = txt.get_window_extent(renderer=renderer)
        w = bb.width / fig.bbox.width + 0.024
        box = FancyBboxPatch(
            (x, y - line_h / 2 + 0.015), w, line_h - 0.03,
            boxstyle="round,pad=0.004", facecolor=color, edgecolor="none",
            transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(box)
        x += w + 0.012
        if x > 0.9 or pc["text"].endswith("\n"):
            x = 0.01
            y -= line_h

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    tok = load_tokenizer()
    text, pieces = dissect(tok)

    print("=== applied chat template (raw string) ===")
    print(text)
    print(f"=== {len(pieces)} tokens (special tokens marked *) ===")
    for pc in pieces:
        disp = pc["text"].replace("\n", "\\n")
        mark = " *" if pc["special"] else "  "
        print(f'{mark} {pc["id"]:>6}  {disp!r}')

    if args.figure:
        FIG_DIR.mkdir(exist_ok=True)
        fig_chat_template(tok, FIG_DIR / "chat_template.png")


if __name__ == "__main__":
    main()
