"""Graphical deliverables from the trained tiny-GPT.

Produces (into figures/):
  loss_curve.png                 train/val cross-entropy over training
  attention_heads.png            layer x head attention grid for a prompt
  generation_over_training.png   same prompt sampled at 0/25/50/100 % of training
  positional_embeddings.png      the learned position-embedding matrix as a heatmap
  attention_head_evolution.gif   one head's attention pattern across checkpoints

Run:  uv run python scripts/figures.py   (after scripts/train.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data import CharTokenizer
from src.model import GPT, GPTConfig


def load_final():
    ckpt = torch.load(C.MODELS / "ckpt_final.pt", map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.load(C.MODELS / "tokenizer_chars.json")
    return model, cfg, tok


# --------------------------------------------------------------------------
def fig_loss_curve(log):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(log["iters"], log["train"], label="train", lw=2)
    ax.plot(log["iters"], log["val"], label="val", lw=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title("Tiny-GPT on tiny-shakespeare — loss descent")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = C.FIGURES / "loss_curve.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
def fig_attention_heads(model, cfg, tok):
    prompt = "ROMEO: But soft, what light"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        model(ids)
    toks = list(prompt)
    T = len(toks)

    fig, axes = plt.subplots(
        cfg.n_layer, cfg.n_head,
        figsize=(2.1 * cfg.n_head, 2.1 * cfg.n_layer),
        squeeze=False,
    )
    for li in range(cfg.n_layer):
        att = model.blocks[li].attn.last_attn[0]   # (n_head, T, T)
        for hi in range(cfg.n_head):
            ax = axes[li][hi]
            ax.imshow(att[hi].numpy(), cmap="magma", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if li == 0:
                ax.set_title(f"head {hi}", fontsize=9)
            if hi == 0:
                ax.set_ylabel(f"layer {li}", fontsize=9)
    fig.suptitle(
        f'Per-head causal attention for "{prompt}"\n'
        "bright = query (row) attends to key (col); note different heads specialize",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = C.FIGURES / "attention_heads.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
def _wrap_sample(text: str, width: int = 34) -> str:
    """Hard-wrap each already-newlined line to `width` chars for tidy panels."""
    import textwrap
    out = []
    for line in text.split("\n"):
        out.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(out[:34])


def fig_generation_over_training(log):
    cps = log["checkpoints"]
    fig, axes = plt.subplots(1, len(cps), figsize=(3.2 * len(cps), 5.6))
    if len(cps) == 1:
        axes = [axes]
    total = C.MAX_ITERS
    for ax, cp in zip(axes, cps):
        pct = round(100 * cp["iter"] / total)
        ax.axis("off")
        ax.set_title(f"iter {cp['iter']}  ({pct}%)", fontsize=10, fontweight="bold")
        text = _wrap_sample(cp["sample"])
        ax.text(
            0.02, 0.99, text, family="monospace", fontsize=6.6, va="top", ha="left",
            linespacing=1.25, transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="#f4f1e8", ec="#bbb", pad=0.5),
        )
    fig.suptitle(
        'Watching it learn — same prompt "ROMEO:" sampled during training\n'
        "noise → spelling → words → Shakespeare-ish lines",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = C.FIGURES / "generation_over_training.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
def fig_positional_embeddings(model, cfg):
    wpe = model.wpe.weight.detach().numpy()      # (block_size, d_model)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(wpe, cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("position in context")
    ax.set_title(
        "Learned positional-embedding matrix (block_size × d_model)\n"
        "structured bands = the model has learned a notion of 'where'",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = C.FIGURES / "positional_embeddings.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
def gif_attention_evolution(log):
    cps = log["checkpoints"]
    prompt = C.SAMPLE_PROMPT
    mats = [np.array(cp["attn_l0h0"]) for cp in cps]
    T = mats[0].shape[0]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mats[0], cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_xticklabels(list(prompt), fontsize=8)
    ax.set_yticklabels(list(prompt), fontsize=8)
    title = ax.set_title("")

    def update(f):
        im.set_data(mats[f])
        pct = round(100 * cps[f]["iter"] / C.MAX_ITERS)
        title.set_text(f"layer 0, head 0 — iter {cps[f]['iter']} ({pct}%)")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(mats), interval=800, blit=False)
    out = C.FIGURES / "attention_head_evolution.gif"
    anim.save(out, writer=animation.PillowWriter(fps=1.4))
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    log = json.loads((C.MODELS / "train_log.json").read_text())
    model, cfg, tok = load_final()
    fig_loss_curve(log)
    fig_attention_heads(model, cfg, tok)
    fig_generation_over_training(log)
    fig_positional_embeddings(model, cfg)
    gif_attention_evolution(log)


if __name__ == "__main__":
    main()
