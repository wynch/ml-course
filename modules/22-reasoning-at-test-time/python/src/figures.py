"""Redraw every figure from results/analysis.json; no model is loaded.

    uv run python src/figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURES, read_jsonl, GEN, read_results

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, DARK = "#8a8a85", "#3a3a38"
TITLE = "Qwen3-1.7B, 4-bit, thinking off · 200 GSM8K test problems"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e4e0", lw=0.8)
    ax.set_axisbelow(True)


def fig_cot(a: dict) -> None:
    s1 = a["section1"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    names = ["answer only", "chain of thought"]
    vals = [s1["direct_acc"], s1["greedy_acc"]]
    ax.bar(names, vals, color=[GREY, BLUE], width=0.55)
    for x, v, t in zip(range(2), vals, [s1["direct_mean_tokens"], s1["greedy_mean_tokens"]]):
        ax.text(x, v + 0.015, f"{v:.1%}\n{t:.0f} tokens", ha="center", va="bottom", fontsize=9, color=DARK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy (greedy)")
    _style(ax)
    toks = [r["tokens"] for r in read_jsonl(GEN / "greedy.jsonl")]
    bx.hist(toks, bins=range(0, 264, 8), color=BLUE)
    bx.axvline(256, color=DARK, lw=1, ls="--")
    bx.text(250, bx.get_ylim()[1] * 0.92, "cap 256", ha="right", fontsize=8, color=DARK)
    bx.set_xlabel("tokens in the greedy chain of thought")
    bx.set_ylabel("problems")
    _style(bx)
    fig.suptitle(TITLE, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "cot.png", dpi=130)
    plt.close(fig)


def fig_vote(a: dict) -> None:
    s1, s2, s3 = a["section1"], a["section2"], a["section3"]
    ns = list(range(1, a["n_samples"] + 1))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(s1["greedy_acc"], color=DARK, lw=1, ls="--")
    ax.text(8.1, s1["greedy_acc"], f"greedy {s1['greedy_acc']:.1%}", va="center", fontsize=8, color=DARK)
    single = s2["single_sample_acc"]
    ax.fill_between([1, 8], min(single), max(single), color=GREY, alpha=0.18, lw=0)
    ax.text(1.05, min(single) + 0.003, "one sample: range over the 8 draws", va="bottom", fontsize=8, color=DARK)
    ax.plot(ns, [s3["pass_at_n_mean_over_subsets"][str(n)] for n in ns], "o:", color=YELLOW, lw=2, ms=6,
            label="any of n right (ceiling)")
    ax.plot(ns, [s2["vote_mean_over_subsets"][str(n)] for n in ns], "o-", color=BLUE, lw=2, ms=6,
            label="majority vote, mean over all subsets of n")
    ax.plot(ns, [s2["vote_prefix"][str(n)] for n in ns], "x", color=BLUE, ms=7,
            label="majority vote, samples 1..n")
    ax.set_xlabel("samples per problem, n")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0.7, 9.4)
    ax.set_xticks(ns)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.set_title(TITLE, fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "vote.png", dpi=130)
    plt.close(fig)


def fig_cost(a: dict) -> None:
    c, s3 = a["cost"], a["section3"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax.plot([p["tokens"] for p in c["vote"]], [p["acc"] for p in c["vote"]], "o-", color=BLUE, lw=2,
            ms=6, label="majority vote")
    ax.plot([p["tokens"] for p in c["best_of_n"]], [p["acc"] for p in c["best_of_n"]], "s-", color=ORANGE,
            lw=2, ms=6, label="best-of-n, self-graded")
    wv = s3["weighted_vote_mean_over_subsets"]
    ax.plot([p["tokens"] for p in c["vote"]], [wv[str(p["n"])] for p in c["vote"]], "^-", color=AQUA,
            lw=2, ms=6, label="vote weighted by self-grade")
    ax.plot([c["greedy"]["tokens"]], [c["greedy"]["acc"]], "D", color=DARK, ms=8, label="greedy chain of thought")
    ax.plot([c["direct"]["tokens"]], [c["direct"]["acc"]], "D", color=GREY, ms=8, label="answer only")
    for p in c["vote"]:
        if p["n"] in (1, 4, 8):
            ax.annotate(f"n={p['n']}", (p["tokens"], p["acc"]), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=DARK)
    ax.set_xscale("log")
    ax.set_xlabel("tokens generated per problem (log scale)")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    _style(ax)
    a_pp = a["per_problem"]
    gold = {p["id"]: p["gold"] for p in a_pp}
    right, wrong = [], []
    for p in a_pp:
        for ans, s in zip(p["samples"], p["logodds"]):
            (right if ans is not None and abs(float(ans) - float(gold[p["id"]])) < 1e-6 else wrong).append(s)
    lo_, hi_ = min(right + wrong), max(right + wrong)
    bins = [lo_ + (hi_ - lo_) * i / 30 for i in range(31)]
    bx.hist(right, bins=bins, color=BLUE, alpha=0.75, label=f"right samples ({len(right)})")
    bx.hist(wrong, bins=bins, color=ORANGE, alpha=0.75, label=f"wrong samples ({len(wrong)})")
    bx.set_xlabel("self-grade: log P(Yes) − log P(No)")
    bx.axvline(0, color=DARK, lw=1, ls="--")
    bx.set_ylabel("samples")
    bx.set_yscale("log")
    bx.legend(fontsize=8, frameon=False, loc="upper center")
    _style(bx)
    fig.suptitle(TITLE, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "cost.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    a = read_results("analysis")
    fig_cot(a)
    fig_vote(a)
    fig_cost(a)
    print("wrote figures/cot.png, figures/vote.png, figures/cost.png")


if __name__ == "__main__":
    main()
