"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

To recompute the numbers too, run each script with --figures instead.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, read_results

COL = {"bm25": "#9a6a00", "minilm": "#1f918d", "smollm2": "#453781", "gold": "#5d6b67", "none": "#b7410e"}
NAME = {"bm25": "BM25", "minilm": "MiniLM cosine", "smollm2": "SmolLM2 mean-pool cosine",
        "gold": "gold paragraph (ceiling)", "none": "no passage"}


def _save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def fig_embed() -> None:
    r = read_results("embed")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    lo, hi = r["models"]["minilm"]["hist"]["range"]
    for ax, name in zip(axes, ("minilm", "smollm2")):
        m = r["models"][name]
        h = m["hist"]
        edges = np.linspace(lo, hi, len(h["gold"]) + 1)
        ax.stairs(h["random"], edges, fill=True, alpha=.35, color="#5d6b67", label="question vs a random paragraph")
        ax.stairs(h["gold"], edges, fill=True, alpha=.55, color=COL[name], label="question vs its own paragraph")
        ax.set_title(f"{NAME[name]}: gold beats random {m['gold_beats_random']:.1%}", fontsize=10)
        ax.set_xlabel("cosine")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("questions (of 10,570)")
    fig.suptitle("Cosine between a question and a paragraph, SQuAD validation", fontsize=11)
    _save(fig, "embed.png")


def fig_retrieval() -> None:
    r = read_results("retrieval")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ks = np.arange(1, 21)
    ax.plot(ks, np.ones_like(ks), color=COL["gold"], ls="--", lw=1.2, label=NAME["gold"])
    for n in ("bm25", "minilm", "smollm2"):
        ax.plot(ks, r["curves"][n], marker="o", ms=3, color=COL[n], label=NAME[n])
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k (paragraphs returned)")
    ax.set_ylabel("recall@k")
    ax.set_title("Is the gold paragraph in the top k? 10,570 questions, 2,067 paragraphs", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "retrieval.png")


def fig_rag() -> None:
    r = read_results("rag")
    conds = r["conditions"]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    em = [r["table"][c]["em"] for c in conds]
    f1 = [r["table"][c]["f1"] for c in conds]
    se = [r["table"][c]["em_se"] for c in conds]
    ax.bar(x - .2, em, .38, yerr=se, capsize=3, color=[COL[c] for c in conds], label="exact match (± 1 s.e.)")
    ax.bar(x + .2, f1, .38, color=[COL[c] for c in conds], alpha=.45, label="F1")
    for i, (a, b) in enumerate(zip(em, f1)):
        ax.text(i - .2, a + .03, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + .2, b + .03, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["no passage", "BM25 top-1", "MiniLM top-1", "gold (ceiling)"])
    ax.set_ylim(0, 1)
    ax.set_title(f"SmolLM2-360M-Instruct on {r['n']} SQuAD questions", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "rag.png")


def main() -> None:
    fig_embed()
    fig_retrieval()
    fig_rag()


if __name__ == "__main__":
    main()
