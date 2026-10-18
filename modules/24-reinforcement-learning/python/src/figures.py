"""Redraw every README figure from the committed results/*.json, no training needed.

    uv run python src/figures.py

To recompute the numbers too, run each script with --figures instead.
Every band is the full range (min to max) over the seeds; the line is the median.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from common import FIGURES, RESULTS, read_results  # noqa: E402

# Fixed categorical order: each entity keeps its colour in every figure.
C = {"egreedy": "#2a78d6", "ucb": "#eb6834", "thompson": "#1baf7a", "random": "#8a9894",
     "reinforce": "#2a78d6", "ppo": "#1baf7a", "q": "#2a78d6"}
LABEL = {"egreedy": "ε-greedy (ε 0.1)", "ucb": "UCB1", "thompson": "Thompson", "random": "random",
         "reinforce": "REINFORCE", "ppo": "PPO"}
INK, MUTED, GRID = "#1c2422", "#5c6b67", "#e3e8e6"

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.8, "figure.dpi": 130, "savefig.bbox": "tight",
})


def _kfmt(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))


def _save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def _band(ax, x, rows, color, label):
    a = np.array(rows, dtype=float)
    ax.fill_between(x, np.nanmin(a, 0), np.nanmax(a, 0), color=color, alpha=0.18, lw=0)
    ax.plot(x, np.nanmedian(a, 0), color=color, label=label)


def fig_bandits():
    r = read_results("bandits")
    g = np.array(r["grid"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, xlog in zip(axes, (False, True)):
        for rule in ("random", "egreedy", "ucb", "thompson"):
            _band(ax, g, r["rules"][rule]["regret"], C[rule], f"{LABEL[rule]}: {r['rules'][rule]['final_median']:,.0f}")
        if xlog:
            gl = g[g >= 10]
            ax.plot(gl, r["lai_robbins_constant"] * np.log(gl), color=INK, ls="--", lw=1.1,
                    label=f"Lai–Robbins rate, {r['lai_robbins_constant']:.1f}·ln t (holds as t → ∞)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(0.5, 5000)
            ax.set_title("Same, log–log: linear regret is a slope-1 line", fontsize=10, loc="left")
        else:
            ax.set_ylim(0, 600)
            ax.set_title(f"Regret over {r['pulls']:,} pulls, {r['seeds']} seeds (random leaves the chart)",
                         fontsize=10, loc="left")
        ax.set_xlabel("pull")
        ax.set_ylabel("regret (pulls lost to the best arm)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "bandits.png")


def fig_qlearning():
    r = read_results("qlearning")
    steps = np.array([x["steps"] for x in r["runs"]])
    ep = np.arange(1, r["episodes"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    _band(ax, ep, steps, C["q"], f"Q-learning, {r['seeds']} seeds")
    ax.axhline(r["bfs_shortest"], color=INK, ls="--", lw=1.1, label=f"shortest path (BFS): {r['bfs_shortest']}")
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel("moves to reach the goal")
    ax.set_title(f"Moves per episode (ε {r['epsilon']}, α {r['alpha']})", fontsize=10, loc="left")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    n = r["size"]
    V = np.array(r["runs"][0]["V"], dtype=float)
    for w in r["walls"]:
        V[w[0], w[1]] = np.nan
    im = ax.imshow(V, cmap="viridis", origin="upper")
    p = np.array(r["runs"][0]["path"])
    ax.plot(p[:, 1], p[:, 0], color="white", lw=2)
    ax.plot(*r["start"][::-1], "o", color="white", ms=7)
    ax.plot(*r["goal"][::-1], "*", color="white", ms=12)
    for w in r["walls"]:
        ax.add_patch(plt.Rectangle((w[1] - 0.5, w[0] - 0.5), 1, 1, color="#3a3a3a"))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(False)
    ax.set_title("Seed 0: max Q per cell and the greedy path", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="max Q (= − moves to goal when right)")
    fig.tight_layout()
    _save(fig, "qlearning.png")


PPO_RUNS = (("ppo", "PPO, lr 2.5e-4 (CleanRL)", "#1baf7a"), ("ppo_fast", "PPO, lr 2.5e-3", "#7a4fd6"),
            ("ppo_fast_noclip", "PPO, lr 2.5e-3, no clip", "#d6457a"))


def _rolling(x, k=10):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def _curves(r):
    return [[np.nan if v is None else v for v in run["curve"]] for run in r["runs"]]


def fig_cartpole():
    rf, pp = read_results("reinforce"), read_results("ppo")
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for name, r in (("reinforce", rf), ("ppo", pp)):
        _band(ax, np.array(r["grid"]), _curves(r), C[name], f"{LABEL[name]}, {len(r['runs'])} seeds")
    ax.axhline(500, color=INK, ls="--", lw=1.0, label="500: the episode limit")
    ax.set_ylim(0, 520)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("return (mean of the last 20 episodes)")
    ax.set_title("CartPole: return over training (line: median; band: min to max of 3 seeds)", fontsize=10, loc="left")
    ax.legend(fontsize=8.5, loc="lower right")
    _kfmt(ax)
    fig.tight_layout()
    _save(fig, "cartpole.png")


def fig_ppo_clip():
    runs = [(read_results(n), lab, col) for n, lab, col in PPO_RUNS]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    ax = axes[0]
    for r, lab, col in runs:
        _band(ax, np.array(r["grid"]), _curves(r), col, lab)
    ax.axhline(500, color=INK, ls="--", lw=1.0)
    ax.set_ylim(0, 520)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("return (mean of the last 20 episodes)")
    ax.set_title("Return, 3 seeds each", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _kfmt(ax)

    ax = axes[1]
    for r, lab, col in runs:
        st = np.array([x["step"] for x in r["runs"][0]["log"]])
        rows = [_rolling([x["clip_frac"] for x in run["log"]]) for run in r["runs"]]
        _band(ax, st, rows, col, lab)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("share of samples with |ratio − 1| > 0.2")
    ax.set_title("Clip fraction (mean of 10 updates), 3 seeds", fontsize=10, loc="left")
    _kfmt(ax)

    ax = axes[2]
    for r, lab, col in runs[1:]:
        bins = np.array(r["ratio_bins"])
        mid = (bins[:-1] + bins[1:]) / 2
        log0 = r["runs"][0]["log"]
        h = np.sum([np.array(x["ratio_hist_last_epoch"], float) for x in log0[:100]], axis=0)
        ax.plot(mid, h / h.sum(), color=col, lw=1.6, label=lab)
    ax.axvspan(0.8, 1.2, color=GRID, alpha=0.6, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("ratio π_new / π_old in the last epoch (ends: 0.5 or less, 1.5 or more)")
    ax.set_ylabel("share of samples")
    ax.set_title("Seed 0, updates 1–100 (shaded: inside the clip)", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "ppo_clip.png")


def main() -> None:
    have = {p.stem for p in RESULTS.glob("*.json")}
    if "bandits" in have:
        fig_bandits()
    if "qlearning" in have:
        fig_qlearning()
    if {"reinforce", "ppo"} <= have:
        fig_cartpole()
    if {"ppo", "ppo_fast", "ppo_fast_noclip"} <= have:
        fig_ppo_clip()


if __name__ == "__main__":
    main()
