"""The module's figures, from the committed results; no model is loaded.

    uv run python src/figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analyse  # noqa: E402
import common  # noqa: E402
import optimise  # noqa: E402

COLOURS = {"ape": "#1f77b4", "opro": "#ff7f0e", "textgrad": "#2ca02c", "gepa": "#9467bd",
           "dspy": "#8c564b"}
NAMES = {"ape": "APE", "opro": "OPRO", "textgrad": "TextGrad-style", "gepa": "GEPA-style",
         "dspy": "DSPy GEPA"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def curves(a, runs):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m, rs in runs.items():
        for r in rs:
            c = np.array(r["curve"])
            x = np.concatenate([c[:, 0], [optimise.BUDGET]])
            y = np.concatenate([c[:, 1], [c[-1, 1]]])
            ax.step(x, 100 * y, where="post", color=COLOURS[m], alpha=0.85, lw=1.4,
                    label=NAMES[m] if r["seed"] == 0 else None)
    ax.axhline(100 * a["baseline"]["train"], color="grey", ls="--", lw=1)
    ax.text(40, 100 * a["baseline"]["train"] + 0.4, "baseline", ha="left",
            va="bottom", color="grey", fontsize=9)
    ax.set_xlabel("training questions scored (budget spent)")
    ax.set_ylabel("best training accuracy so far (%)")
    ax.set_title("Training accuracy against budget, 3 seeds per optimiser")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "curves.png", dpi=150)
    plt.close(fig)


def splits(a):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.0), sharey=True)
    order = [m for m in optimise.METHODS if m in a["methods"]]
    for ax, split in zip(axes, ("train", "heldout", "test")):
        for i, m in enumerate(order):
            vals = 100 * np.array(a["methods"][m][split]["values"])
            ax.scatter(np.full(len(vals), i), vals, color=COLOURS[m], s=28, zorder=3)
            ax.hlines(vals.mean(), i - 0.3, i + 0.3, color=COLOURS[m], lw=2)
        extra = len(order)
        if "dspy" in a and split in a["dspy"]:
            ax.scatter([extra], [100 * a["dspy"][split]], color=COLOURS["dspy"], marker="D", s=34, zorder=3)
        ax.axhline(100 * a["baseline"][split], color="grey", ls="--", lw=1, label="baseline")
        ax.axhline(100 * a["by_hand"][split], color="black", ls=":", lw=1, label="written by hand")
        ax.set_xticks(range(extra + 1), [NAMES[m] for m in order] + [NAMES["dspy"]], rotation=35,
                      ha="right", fontsize=8.5)
        ax.set_title({"train": "train (60, seen)", "heldout": "held-out (300, same source)",
                      "test": "TREC test (500)"}[split], fontsize=10)
        style(ax)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("The chosen instruction of each run on the three splits", fontsize=11)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "splits.png", dpi=150)
    plt.close(fig)


def overfit(a):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for r in a["runs"]:
        ax.scatter(100 * r["train"], 100 * r["heldout"], color=COLOURS[r["method"]], s=34,
                   label=NAMES[r["method"]] if r["seed"] == 0 else None)
    ax.scatter(100 * a["baseline"]["train"], 100 * a["baseline"]["heldout"], color="grey",
               marker="s", s=40, label="baseline")
    ax.scatter(100 * a["by_hand"]["train"], 100 * a["by_hand"]["heldout"], color="black",
               marker="x", s=40, label="written by hand")
    lo, hi = 50, 90
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8)
    ax.text(84, 81.5, "held-out = train", ha="center", fontsize=8, color="grey", rotation=42)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("train accuracy (%)")
    ax.set_ylabel("held-out accuracy (%)")
    ax.set_title("Below the line: what the train score overstates")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "overfit.png", dpi=150)
    plt.close(fig)


def steps(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for r in runs.get("textgrad", []):
        tr = r["trace"]
        cur = []
        c = tr[0]["acc"]
        for t in tr:
            if t["accepted"]:
                c = t["acc"]
            cur.append(c)
        x = np.arange(len(tr))
        ax.plot(x, 100 * np.array(cur), color=COLOURS["textgrad"], lw=1.2, alpha=0.8)
        acc = np.array([t["acc"] for t in tr])
        ok = np.array([t["accepted"] for t in tr])
        ax.scatter(x[ok], 100 * acc[ok], color=COLOURS["textgrad"], s=12)
        ax.scatter(x[~ok], 100 * acc[~ok], facecolor="none", edgecolor="#c0504d", s=14)
    ax.set_xlabel("TextGrad-style step")
    ax.set_ylabel("training accuracy (%)")
    ax.set_title("Each edit: kept (filled) or undone (red ring)", fontsize=10)
    style(ax)
    ax = axes[1]
    for r in runs.get("gepa", []):
        tr = r["trace"]
        x = np.arange(len(tr))
        d = np.array([t["mini_after"] - t["mini_before"] for t in tr])
        kept = np.array([t["kept"] for t in tr])
        ax.scatter(x[kept], d[kept] + 0.08 * r["seed"], color=COLOURS["gepa"], s=12)
        ax.scatter(x[~kept], d[~kept] + 0.08 * r["seed"], color="#bbbbbb", s=8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("GEPA-style child")
    ax.set_ylabel("minibatch: child right − parent right")
    ax.set_title("A child goes to the full pass only if it wins its 6 questions", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "steps.png", dpi=150)
    plt.close(fig)


def main():
    common.FIGURES.mkdir(exist_ok=True)
    a = common.read_results("analysis")
    runs = analyse.load_runs()
    curves(a, runs)
    splits(a)
    overfit(a)
    steps(runs)
    print("figures written")


if __name__ == "__main__":
    main()
