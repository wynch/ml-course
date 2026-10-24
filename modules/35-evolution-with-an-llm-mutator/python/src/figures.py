"""Draw the module's figures from results/analysis.json.

    uv run python src/figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
import task  # noqa: E402

COLORS = {"evolve": "#1d6fb8", "noselect": "#c2410c", "random": "#7e22ce",
          "noverifier": "#b3261e", "evolve_flat": "#15803d"}
LABELS = {"evolve": "evolution (islands, by score)", "noselect": "LLM, no selection",
          "random": "random expressions, no LLM", "noverifier": "no verifier",
          "evolve_flat": "evolution, parents uniform"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def curves(a: dict) -> None:
    bf, ff = a["baselines"]["best_fit"]["evolve"], a["baselines"]["first_fit"]["evolve"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for arm in ("random", "noselect", "evolve_flat", "evolve"):
        runs = [r for r in a["runs"].values() if r["arm"] == arm]
        for i, r in enumerate(runs):
            ax.plot(np.arange(1, len(r["best_so_far"]) + 1), r["best_so_far"], color=COLORS[arm],
                    lw=1.8, alpha=0.85, label=LABELS[arm] if i == 0 else None)
    ax.axhline(bf, color="0.3", ls="--", lw=1)
    ax.text(5, bf + 0.03, f"best fit {bf:.2f}%", color="0.3", fontsize=9)
    ax.set_ylim(min(min(r["best_so_far"]) for r in a["runs"].values() if r["arm"] != "noverifier") - 0.15,
                bf + 0.25)
    ax.set_xlabel("candidates sampled (same budget for every arm)")
    ax.set_ylabel("best excess over L1 so far, evolve set (%)")
    ax.set_title("Best program found so far, one line per seed")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "curves.png", dpi=150)
    plt.close(fig)
    print(f"first fit {ff}")


def heldout(a: dict) -> None:
    sets = ["evolve", "heldout", "shift"]
    titles = ["evolve set (seen)", "held-out (same sizes)", "shift (other sizes)"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=False)
    arms = [arm for arm in ("random", "noselect", "evolve_flat", "evolve", "noverifier")
            if arm in a["summary"]]
    for ax, s, title in zip(axes, sets, titles):
        bf = a["baselines"]["best_fit"][s]
        for x, arm in enumerate(arms):
            vals = [bf - r["kept"]["excess"][s] for r in a["runs"].values()
                    if r["arm"] == arm and r["kept"]["excess"]]
            ax.scatter([x] * len(vals), vals, color=COLORS[arm], s=40, zorder=3)
            if vals:
                ax.hlines(np.mean(vals), x - 0.3, x + 0.3, color=COLORS[arm], lw=2)
        ax.axhline(0, color="0.3", ls="--", lw=1)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a_.replace("_", "\n") for a_ in arms], fontsize=8)
        ax.set_title(title, fontsize=10)
        style(ax)
    axes[0].set_ylabel("points better than best fit\n(best fit excess minus kept program's)")
    fig.suptitle("The kept program of each run, on the set it was chosen on and two it never saw",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "heldout.png", dpi=150)
    plt.close(fig)


def noverifier(a: dict) -> None:
    runs = [r for r in a["runs"].values() if r["arm"] == "noverifier"]
    if not runs:
        return
    bf = a["baselines"]["best_fit"]["evolve"]
    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 1.9 * len(runs) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, runs):
        ex = r["excess_trace"]
        xs = [i + 1 for i, e in enumerate(ex) if e is not None]
        ys = [min(e, 20) for e in ex if e is not None]
        ax.scatter(xs, ys, s=6, color=COLORS["noverifier"], label="child runs (score hidden from loop)")
        fails = [i + 1 for i, s in enumerate(r["status_trace"]) if s != "valid"]
        ax.scatter(fails, [20.8] * len(fails), marker="|", color="0.2", s=30, label="child fails to run")
        ax.axhline(bf, color="0.3", ls="--", lw=1)
        ax.set_ylim(3, 22.5)
        ax.set_ylabel(f"seed {r['seed']}\nexcess (%)", fontsize=9)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    axes[-1].set_xlabel("child number (each child becomes the next parent)")
    fig.suptitle("No verifier: every child is accepted; scores shown here, capped at 20%, were never seen",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "noverifier.png", dpi=150)
    plt.close(fig)


def status(a: dict) -> None:
    kinds = ["valid", "duplicate", "invalid", "crash", "screen", "timeout", "noparse"]
    colors = ["#1b7a3a", "#9aa5b1", "#e8a400", "#b3261e", "#7e22ce", "#000000", "#c2410c"]
    arms = [arm for arm in ("evolve", "evolve_flat", "noselect", "noverifier", "random")
            if arm in a["summary"]]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for y, arm in enumerate(arms):
        runs = [r for r in a["runs"].values() if r["arm"] == arm]
        n = sum(r["candidates"] for r in runs)
        left = 0.0
        for k, c in zip(kinds, colors):
            share = sum(r["status"].get(k, 0) for r in runs) / n
            ax.barh(y, share, left=left, color=c, label=k if y == 0 else None)
            left += share
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([LABELS[a_] for a_ in arms], fontsize=9)
    ax.set_xlabel("share of sampled candidates, all seeds")
    ax.legend(frameon=False, fontsize=8, ncol=7, loc="upper center", bbox_to_anchor=(0.45, 1.2))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(common.FIGURES / "status.png", dpi=150)
    plt.close(fig)


def fill(a: dict) -> None:
    """Bins saved against best fit on every instance the loop never saw."""
    kept = [(f"{r['arm']} s{r['seed']}", r["kept"]["code"], COLORS[r["arm"]])
            for r in a["runs"].values() if r["arm"] in ("evolve", "random") and r["kept"]["excess"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    for ax, name in zip(axes, ("heldout", "shift")):
        insts = task.instances(name)
        bf = [len(task.pack(s_, c, task.best_fit)) for s_, c in insts]
        width = 0.8 / len(kept)
        for k, (label, code, color) in enumerate(kept):
            env = {"np": np}
            exec(code, env)
            with np.errstate(all="ignore"):
                used = [len(task.pack(s_, c, env["priority"])) for s_, c in insts]
            x = np.arange(len(insts)) + (k - (len(kept) - 1) / 2) * width
            ax.bar(x, np.array(bf) - np.array(used), width, color=color,
                   alpha=0.45 + 0.55 * (k % 3) / 2, label=label if name == "heldout" else None)
        ax.axhline(0, color="0.3", lw=1)
        ax.set_xticks(range(len(insts)))
        ax.set_xlabel(f"{'held-out' if name == 'heldout' else 'shift'} instance")
        ax.set_title("same sizes as the evolve set" if name == "heldout" else "other sizes, bins of 150", fontsize=10)
        style(ax)
    axes[0].set_ylabel("bins saved against best fit")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    fig.suptitle("Each run's kept program on instances it never saw (above 0: fewer bins than best fit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fill.png", dpi=150)
    plt.close(fig)


def main() -> None:
    common.FIGURES.mkdir(exist_ok=True)
    a = common.read_results("analysis")
    curves(a)
    heldout(a)
    noverifier(a)
    status(a)
    fill(a)
    print("figures written")


if __name__ == "__main__":
    main()
