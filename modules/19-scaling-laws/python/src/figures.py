"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

Sizes 32 and 48 ran two seeds; wherever they appear, the shaded band spans the
two seeds and the line is seed 0.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, MARKED_STEPS, SIZES, STEPS, TWO_SEEDS, WARMUP, read_results, read_runs

COLORS = {d: plt.cm.viridis(x) for d, x in zip(SIZES, np.linspace(0.82, 0.05, len(SIZES)))}


def _label(d: int, runs: dict) -> str:
    n = next(r for r in runs[d]["runs"] if r["seed"] == 0)["n_non_embedding"]
    return f"d {d}, N {n / 1e6:.2f}M" if n >= 1e6 else f"d {d}, N {n / 1e3:.0f}k"


def _curve(run: dict, after_warmup: bool = False):
    pts = [(c["tokens"], c["held_loss"]) for c in run["curve"] if not after_warmup or c["step"] > WARMUP]
    return np.array(pts).T


def _band(ax, d: int, runs: dict, x_of=lambda t, n: t, after_warmup: bool = False) -> None:
    """Shade between the two seeds of a two-seed size."""
    if d not in TWO_SEEDS:
        return
    rs = runs[d]["runs"]
    (t0, l0), (_, l1) = (_curve(r, after_warmup) for r in rs)
    n = rs[0]["n_non_embedding"]
    ax.fill_between(x_of(t0, n), np.minimum(l0, l1), np.maximum(l0, l1), color=COLORS[d], alpha=0.35, lw=0)


def fig_curves(path=FIGURES / "loss_curves.png") -> None:
    runs = read_runs()
    try:
        base = read_results("fit")["baselines"]
    except FileNotFoundError:            # train.py --figures before fit.py has run
        base = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for d in SIZES:
        t, l = _curve(next(r for r in runs[d]["runs"] if r["seed"] == 0))
        _band(ax, d, runs)
        ax.plot(t, l, "o-", ms=3, lw=1.6, color=COLORS[d], label=_label(d, runs))
    for k, ls in (("bigram", "--"), ("unigram", ":")):
        if k in base:
            ax.axhline(base[k], color="0.45", ls=ls, lw=1, label=f"{k} table: {base[k]:.2f}")
    ax.axvline(WARMUP * 4096, color="0.75", lw=1)
    ax.text(WARMUP * 4096 * 1.05, 4.6, "end of warmup", fontsize=8, color="0.45")
    ax.set(xscale="log", yscale="log", xlabel="training tokens (characters)", ylabel="held-out loss (nats per character)")
    ax.set_yticks([1.8, 2, 2.5, 3, 4, 5])
    ax.set_yticklabels(["1.8", "2", "2.5", "3", "4", "5"])
    ax.minorticks_off()
    ax.set_title("Six char-GPTs, one pass over 4M characters (band: two seeds)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_fit(path=FIGURES / "fit.png") -> None:
    runs = read_runs()
    r = read_results("fit")
    ns = np.array(r["n_non_embedding"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    # left: L against N at 4M tokens
    m = r["l_of_n_main"]
    xs = np.geomspace(ns.min() * 0.8, ns.max() * 1.25, 50)
    for i, d in enumerate(SIZES):
        ls = [c["held_loss"] for rr in runs[d]["runs"] for c in rr["curve"] if c["step"] == STEPS]
        if len(ls) > 1:
            ax[0].vlines(ns[i], min(ls), max(ls), color=COLORS[d], lw=7, alpha=0.45)
        ax[0].plot(ns[i], r["loss_4M_seed0"][i], "o", color=COLORS[d], ms=7, zorder=3)
    ax[0].plot(xs, np.exp(m["a"]) * xs ** -m["exponent"], "--", color="k", lw=1.2,
               label=f"L ∝ N^−{m['exponent']:.3f}, R² {m['r2']:.3f}")
    ax[0].axhline(r["baselines"]["bigram"], color="0.45", ls="--", lw=1, label=f"bigram table: {r['baselines']['bigram']:.2f}")
    ax[0].set(xscale="log", yscale="log", xlabel="non-embedding parameters N",
              ylabel="held-out loss at 4M tokens (nats/char)")
    ax[0].set_yticks([1.8, 2.0, 2.2, 2.4, 2.6])
    ax[0].set_yticklabels(["1.8", "2.0", "2.2", "2.4", "2.6"])
    ax[0].minorticks_off()
    ax[0].set_title("L(N) at 4M tokens (bars: two seeds)", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(alpha=0.25)

    # right: L against D, every size, with the L(D) lines for the largest two
    for d in SIZES:
        t, l = _curve(next(rr for rr in runs[d]["runs"] if rr["seed"] == 0), after_warmup=True)
        _band(ax[1], d, runs, after_warmup=True)
        lw = 2 if d in r["l_of_d_largest_two"] else 1.2
        ax[1].plot(t, l, "o-", ms=3, lw=lw, color=COLORS[d], label=_label(d, runs))
    ts = np.array([c * 4096 for c in r["l_of_d_steps"]], float)
    for d in r["l_of_d_largest_two"]:
        f = r["l_of_d"][str(d)]
        ax[1].plot(ts, np.exp(f["a"]) * ts ** -f["exponent"], "--", color="k", lw=1)
        ax[1].text(ts[-1] * 1.08, np.exp(f["a"]) * ts[-1] ** -f["exponent"],
                   f"d {d}: β {f['exponent']:.3f}\nR² {f['r2']:.2f}", fontsize=8, va="center")
    ax[1].set(xscale="log", yscale="log", xlabel="training tokens D (after warmup)", ylabel="held-out loss (nats/char)")
    ax[1].set_xlim(ts[0] * 0.8, ts[-1] * 2.6)
    ax[1].set_yticks([1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6])
    ax[1].set_yticklabels(["1.8", "2.0", "2.2", "2.4", "2.6", "2.8", "3.2", "3.6"])
    ax[1].minorticks_off()
    ax[1].set_title("L(D), 0.5M to 4M tokens (band: two seeds)", fontsize=10)
    ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_frontier(path=FIGURES / "frontier.png") -> None:
    runs = read_runs()
    fr = read_results("frontier")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for d in SIZES:
        rr = next(x for x in runs[d]["runs"] if x["seed"] == 0)
        t, l = _curve(rr, after_warmup=True)
        n = rr["n_non_embedding"]
        _band(ax[0], d, runs, x_of=lambda t, n: 6 * n * t, after_warmup=True)
        ax[0].plot(6 * n * t, l, "o-", ms=3, lw=1.2, color=COLORS[d], label=_label(d, runs))
    c = np.array([row["C"] for row in fr["rows"]])
    ax[0].plot(c, [row["loss"] for row in fr["rows"]], "-", color="k", lw=3, alpha=0.25, label="lowest at each C")
    ax[0].set(xscale="log", yscale="log", xlabel="compute C = 6 N D (FLOPs)", ylabel="held-out loss (nats/char)")
    ax[0].set_yticks([1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6])
    ax[0].set_yticklabels(["1.8", "2.0", "2.2", "2.4", "2.6", "2.8", "3.2", "3.6"])
    ax[0].minorticks_off()
    ax[0].set_title("Loss against compute (band: two seeds)", fontsize=10)
    ax[0].legend(frameon=False, fontsize=7.5)
    ax[0].grid(alpha=0.25)

    best_n = np.array([row["best_n"] for row in fr["rows"]])
    seed1_n = [next(x for x in runs[d]["runs"] if x["seed"] == 0)["n_non_embedding"] for d in fr["rows_seed1_small"]]
    ax[1].step(c, best_n, where="mid", color="k", lw=1.5, label="best size, seed 0")
    ax[1].step(c, seed1_n, where="mid", color=COLORS[TWO_SEEDS[0]], lw=3, alpha=0.5,
               label="best size, seed 1 for d 32 and 48")
    for d in SIZES:
        rr = next(x for x in runs[d]["runs"] if x["seed"] == 0)
        n = rr["n_non_embedding"]
        ax[1].axhline(n, color=COLORS[d], lw=0.8, alpha=0.6)
        ax[1].axvline(6 * n * rr["curve"][-1]["tokens"], color=COLORS[d], lw=1, ls=":")
    ax[1].text(0.98, 0.16, "dotted: where each size's run ends (4M tokens)", transform=ax[1].transAxes, bbox=dict(fc="white", ec="none", alpha=0.85),
               ha="right", fontsize=8, color="0.35")
    ax[1].set(xscale="log", yscale="log", xlabel="compute C = 6 N D (FLOPs)", ylabel="best non-embedding N")
    ax[1].set_title("Compute-optimal size among the six", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    fig_curves()
    fig_fit()
    fig_frontier()


if __name__ == "__main__":
    main()
