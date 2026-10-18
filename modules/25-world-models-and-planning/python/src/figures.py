"""Redraw every figure of module 25 from results/*.json; no model is loaded.

    uv run python src/figures.py
"""

from __future__ import annotations

import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

AGENTS = ("bfs", "onehot", "latent", "latent_random", "random")
LABEL = {"bfs": "BFS (ceiling)", "onehot": "one-hot model", "latent": "JEPA latent",
         "latent_random": "JEPA, random weights", "random": "random actions (floor)", "stay_put": "guess: no move",
         "raw_observation": "raw observation"}
COL = {"bfs": "#3fa34d", "onehot": "#1d6fb8", "latent": "#d1495b", "latent_random": "#e0a100",
       "random": "#8a8f8d", "stay_put": "#8a8f8d", "raw_observation": "#3fa34d"}
WALL = "#4f6272"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _board(ax, walls, start=None, goal=None, shade=None, title=None):
    walls = np.asarray(walls)
    n = len(walls)
    ax.imshow(np.zeros((n, n)), cmap=ListedColormap(["#ffffff"]), origin="upper")
    if shade is not None:
        s = np.asarray(shade)
        for r in range(n):
            for c in range(n):
                if s[r, c] > 0 and not walls[r, c] and (start is None or (r, c) != tuple(start)):
                    ax.text(c, r, str(s[r, c]), ha="center", va="center", fontsize=7, color="#5d6b67")
    ax.imshow(np.ma.masked_where(walls == 0, walls), cmap=ListedColormap([WALL]), origin="upper")
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="#d5ddda", lw=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    if start is not None:
        ax.plot(start[1], start[0], "o", color="#1c2422", ms=9)
    if goal is not None:
        ax.plot(goal[1], goal[0], "*", color="#e0a100", ms=15, mec="#1c2422", mew=0.6)
    if title:
        ax.set_title(title, fontsize=9)


def world() -> None:
    W = common.read_results("world")
    ex = W["test_arenas"]["examples"]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    for i, a in enumerate(ex[:3]):
        _board(axs[i], a["walls"], a["start"], a["goal"], a["dist_map"],
               f"test arena {i + 1}: {a['bfs']} steps to the goal")
    ax = axs[3]
    h = np.array(W["raw_arenas"]["bfs_hist_0_20"]) / W["raw_arenas"]["n"]
    x = np.arange(len(h))
    ax.bar(x, h, color=["#1d6fb8" if 5 <= v <= 20 else "#bfc5c3" for v in x])
    ax.set_xlabel("BFS steps, start to goal")
    ax.set_ylabel("share of 10,000 arenas")
    ax.set_title(f"10,000 random arenas; test arenas keep 5-20 (blue).\n"
                 f"Goal unreachable in {1 - W['raw_arenas']['goal_reachable_share']:.1%}", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "world.png", dpi=150)
    plt.close(fig)


def models() -> None:
    R = common.read_results("models")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    lo = R["onehot"]["log"]
    lj = R["jepa"]["log"]
    ax[0].plot([p["step"] for p in lo], [p["loss"] for p in lo], color=COL["onehot"], label="one-hot: cross-entropy")
    ax[0].plot([p["step"] for p in lj], [p["loss"] for p in lj], color=COL["latent"], label="JEPA: latent squared error")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("training step")
    ax[0].set_title("training loss (one batch of 256, every 200 steps)", fontsize=9)
    ax[0].legend(fontsize=8)
    _style(ax[0])
    acc = R["rollout_cell_acc"]
    k = np.arange(1, len(acc["onehot"]) + 1)
    for name in ("onehot", "latent", "latent_random", "stay_put"):
        ax[1].plot(k, acc[name], marker="o", ms=3, color=COL[name], ls="--" if name == "stay_put" else "-",
                   label=LABEL[name])
    ax[1].axhline(1 / 64, color="#8a8f8d", lw=0.6, ls=":")
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("imagined steps")
    ax[1].set_ylabel("imagined cell = real cell")
    ax[1].set_title(f"8 random actions from {R['rollout']['states']:,} new states", fontsize=9)
    ax[1].legend(fontsize=8)
    _style(ax[1])
    P = R["keeps"]["probe_r2"]
    labs = ("agent_cell", "goal_cell", "goal_offset", "wall_count")
    w = 0.26
    for i, f in enumerate(("latent", "latent_random", "raw_observation")):
        v = [P[f][l] for l in labs]
        ax[2].bar(np.arange(len(labs)) + (i - 1) * w, v, w, color=COL[f], label=LABEL[f])
    ax[2].axhline(0, color="#1c2422", lw=0.5)
    ax[2].set_xticks(range(len(labs)), [l.replace("_", " ") for l in labs], fontsize=9)
    ax[2].set_ylim(min(-0.1, min(min(P[f].values()) for f in P) - 0.05), 1.05)
    ax[2].set_ylabel("ridge probe R², held-out")
    ax[2].set_title("what a linear probe reads from each", fontsize=9)
    ax[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    _style(ax[2])
    fig.tight_layout()
    fig.savefig(common.FIGURES / "models.png", dpi=150)
    plt.close(fig)


def plan() -> None:
    P = common.read_results("plan")
    A = P["agents"]
    fig = plt.figure(figsize=(13, 4))
    ax = fig.add_subplot(1, 3, 1)
    y = np.arange(len(AGENTS))[::-1]
    v = [A[a]["success"] for a in AGENTS]
    ax.barh(y, v, color=[COL[a] for a in AGENTS], height=0.6)
    for yi, vi in zip(y, v):
        ax.text(vi + 0.01, yi, f"{vi:.0%}" if vi >= 0.1 else f"{vi:.1%}", va="center", fontsize=9)
    ax.set_yticks(y, [LABEL[a] for a in AGENTS], fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(f"reached the goal within {P['budget']} steps")
    ax.set_title(f"{P['arenas']['n']} test arenas, {P['shots']} × {P['horizon']} random shooting", fontsize=9)
    _style(ax)
    # One arena, the imagined best sequence at the first step for both planners.
    i = next((j for j, e in enumerate(A["onehot"]["episodes"]) if e["success"]), 0)
    ar = P["arenas"]["first"][i]
    for col, name in ((2, "onehot"), (3, "latent")):
        ax = fig.add_subplot(1, 3, col)
        ep = A[name]["episodes"][i]
        _board(ax, ar["walls"], ar["start"], ar["goal"],
               title=f"{LABEL[name]}: {'reached' if ep['success'] else 'missed'} in {ep['steps']} steps")
        path = np.array(ep["path"])
        ax.plot(path[:, 1], path[:, 0], "-", color=COL[name], lw=2.2, alpha=0.8, label="real path")
        im = np.array([ar["start"]] + ep["imagined"][0])
        ax.plot(im[:, 1], im[:, 0], ":o", color="#1c2422", ms=3, lw=1.2, label="imagined, first plan")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "plan.png", dpi=150)
    plt.close(fig)


def main() -> None:
    common.FIGURES.mkdir(exist_ok=True)
    world()
    models()
    plan()
    print("  wrote figures/world.png, models.png, plan.png")


if __name__ == "__main__":
    main()
