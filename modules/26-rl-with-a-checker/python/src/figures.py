"""Redraw every README figure from the committed results/*.json, no training needed.

    uv run python src/figures.py

Bands are the full range (min to max) over the 3 seeds; the line is the mean.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import FIGURES, RESULTS, read_results  # noqa: E402

# Fixed categorical order: each arm keeps its colour in every figure.
C = {"base": "#5c6b67", "grpo": "#2a78d6", "drgrpo": "#eb6834", "rloo": "#1baf7a",
     "random": "#a05fc8", "random_clip": "#d1495b", "direct": "#2a78d6", "worked": "#eb6834"}
LABEL = {"base": "base", "grpo": "GRPO", "drgrpo": "Dr. GRPO", "rloo": "RLOO",
         "random": "random reward", "random_clip": "random reward, 4 clipped steps"}
ARMS = ["grpo", "drgrpo", "rloo", "random", "random_clip"]
SEEDS = [0, 1, 2]
INK, MUTED, GRID = "#1c2422", "#5c6b67", "#e3e8e6"

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.8, "figure.dpi": 130, "savefig.bbox": "tight",
})


def _save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def _runs(arm):
    return [json.loads((RESULTS / "runs" / f"{arm}_s{s}.json").read_text()) for s in SEEDS]


def _band(ax, x, rows, color, label, ls="-"):
    a = np.array(rows, dtype=float)
    ax.fill_between(x, np.nanmin(a, 0), np.nanmax(a, 0), color=color, alpha=0.15, lw=0)
    ax.plot(x, np.nanmean(a, 0), color=color, label=label, ls=ls)


def fig_base():
    s = read_results("sft")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    cv = s["curve_first_200_test"]
    x = [r["step"] for r in cv]
    ax.plot(x, [100 * r["sample_acc_direct"] for r in cv], color=C["direct"], label="direct replies")
    ax.plot(x, [100 * r["sample_acc_worked"] for r in cv], color=C["worked"], label="worked replies")
    ax.plot(x, [100 * r["sample_acc"] for r in cv], color=INK, lw=1.2, ls="--", label="all samples")
    ax.set_xlabel("training step (256 examples each)")
    ax.set_ylabel("right, % (200 test sums, 4 samples each)")
    ax.set_title("Base model: next-character training, stopped at step 3,000", fontsize=10)
    ax.legend(loc="upper left")
    ax = axes[1]
    t = json.loads((RESULTS / "runs" / "base_eval.json").read_text())["final"]
    vals = [100 * t["acc_direct"], 100 * t["acc_worked"]]
    shares = [100 * (1 - t["worked_share"]), 100 * t["worked_share"]]
    bars = ax.bar(["direct\n\"859.\"", "worked\n\"[4+5=9,…]859.\""], vals, color=[C["direct"], C["worked"]], width=0.55)
    for b, v, sh in zip(bars, vals, shares):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}% right\n{sh:.1f}% of samples", ha="center",
                fontsize=8.5, color=INK)
    ax.set_ylim(0, 115)
    ax.set_ylabel("right, % (500 test sums, 128 samples each)")
    ax.set_title("At the stop: two styles, two success rates", fontsize=10)
    _save(fig, "base.png")


def fig_training():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    keys = [("reward", "checker reward on the batch, %", 100), ("worked_share", "worked replies, %", 100),
            ("kl_to_base", "KL to the base, nats per reply", 1), ("length_wrong", "length of wrong replies, characters", 1)]
    for ax, (k, lab, mul) in zip(axes.flat, keys):
        for arm in ARMS:
            runs = _runs(arm)
            x = [r["step"] for r in runs[0]["log"]]
            rows = [[(row[k] if row[k] is not None else np.nan) * mul for row in r["log"]] for r in runs]
            _band(ax, x, rows, C[arm], LABEL[arm], ls="--" if arm.startswith("random") else "-")
        ax.set_ylabel(lab)
        ax.set_xlabel("RL step (32 sums x 8 replies)")
    axes[0, 0].legend(loc="center right", fontsize=8.5)
    fig.suptitle("Training with the checker: what moves, mean and range of 3 seeds", fontsize=11)
    _save(fig, "training.png")


def fig_passk():
    R = read_results("rl")
    base = R["base"]["pass_at_k"]
    ks = [int(k) for k in base]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    ax.plot(ks, [100 * base[str(k)] for k in ks], color=C["base"], marker="o", ms=3, label="base", lw=2.4)
    for arm in ARMS:
        p = R["arms"][arm]["pass_at_k"]
        ax.plot(ks, [100 * p[str(k)]["mean"] for k in ks], color=C[arm], marker="o", ms=3, label=LABEL[arm],
                ls="--" if arm.startswith("random") else "-")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k, tries per sum")
    ax.set_ylabel("pass@k, % of 500 test sums")
    ax.set_title("pass@k after 300 steps, mean of 3 seeds", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    for arm in ARMS:
        d = R["arms"][arm]["pass_at_k_minus_base"]
        lo = [100 * d[str(k)]["min"] for k in ks]
        hi = [100 * d[str(k)]["max"] for k in ks]
        ax.fill_between(ks, lo, hi, color=C[arm], alpha=0.15, lw=0)
        ax.plot(ks, [100 * d[str(k)]["mean"] for k in ks], color=C[arm], marker="o", ms=3,
                ls="--" if arm.startswith("random") else "-")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k, tries per sum")
    ax.set_ylabel("pass@k minus the base's, points")
    ax.set_title("Below 0 would be the crossover", fontsize=10)
    _save(fig, "passk.png")


def fig_library():
    if not (RESULTS / "library_after.json").exists():
        print("  skip figures/library.png: no results/library_after.json yet")
        return
    b, a, t = read_results("library_base"), read_results("library_after"), read_results("library_train")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    hist = [h for h in t["log_history"] if "reward" in h]
    ax.plot([h["step"] for h in hist], [100 * h["reward"] for h in hist], color=C["grpo"])
    ax.set_xlabel("GRPOTrainer step (4 sums x 8 replies)")
    ax.set_ylabel("checker reward on the batch, %")
    ax.set_title("SmolLM2-360M-Instruct + LoRA, TRL", fontsize=10)
    ax = axes[1]
    ks = [int(k) for k in b["measure"]["pass_at_k"]]
    ax.plot(ks, [100 * b["measure"]["pass_at_k"][str(k)] for k in ks], color=C["base"], marker="o", ms=3,
            label="before", lw=2.4)
    ax.plot(ks, [100 * a["measure"]["pass_at_k"][str(k)] for k in ks], color=C["grpo"], marker="o", ms=3,
            label="after")
    ax.axhline(100 * b["measure"]["greedy_acc"], color=C["base"], ls=":", lw=1.2,
               label=f"greedy before, {100 * b['measure']['greedy_acc']:.1f}%")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k, tries per sum (samples at temperature 1)")
    ax.set_ylabel("pass@k, % of 200 test sums")
    ax.set_title("pass@k before and after", fontsize=10)
    ax.legend(loc="lower right")
    _save(fig, "library.png")


def main() -> None:
    fig_base()
    fig_training()
    fig_passk()
    fig_library()


if __name__ == "__main__":
    main()
