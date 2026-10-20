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
C = {"full": "#2a78d6", "full_slow": "#86b3e6", "lora4": "#eb6834", "lora32": "#b0421f",
     "replay": "#1baf7a", "distill": "#a05fc8",
     "backprop": "#d1495b", "cbp": "#1baf7a", "sgd": "#eb6834", "fresh": "#5c6b67",
     "lora8_gentle": "#86b3e6", "lora8": "#2a78d6", "lora64": "#b0421f", "lora8_replay": "#1baf7a"}
LABEL = {"full": "full, rate 1e-3", "full_slow": "full, rate 1e-4", "lora4": "LoRA rank 4", "lora32": "LoRA rank 32",
         "replay": "full + 25% replay", "distill": "full + self-distillation",
         "backprop": "backprop (Adam)", "cbp": "continual backprop (Adam)", "sgd": "backprop (plain SGD)",
         "fresh": "fresh network each task",
         "lora8_gentle": "LoRA r8, rate 1e-4", "lora8": "LoRA r8, rate 1e-3", "lora64": "LoRA r64, rate 1e-3",
         "lora8_replay": "LoRA r8 + chat replay"}
ARMS = ["full", "full_slow", "lora4", "lora32", "replay", "distill"]
PLAST = ["backprop", "cbp", "sgd", "fresh"]
LIB = ["lora8_gentle", "lora8", "lora64", "lora8_replay"]
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


def fig_start():
    s = read_results("start")
    x = [r["step"] for r in s["curve"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(x, [r["old_loss"] for r in s["curve"]], color=INK, label="old text: Shakespeare, held out")
    ax.plot(x, [r["new_loss"] for r in s["curve"]], color=C["full"], ls="--", label="new text: GSM8K solutions, held out")
    ax.set_xlabel("training step on Shakespeare (64 windows of 128 characters)")
    ax.set_ylabel("loss, nats per character")
    ax.set_title(f"The start model: {s['parameters']:,} weights, trained on Shakespeare only", fontsize=10)
    ax.legend()
    _save(fig, "start.png")


def fig_forgetting():
    F = read_results("forget")
    steps = F["steps"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for a in ARMS:
        rs = _runs(a)
        _band(axes[0], steps, [[r["new_loss"] for r in rr["log"]] for rr in rs], C[a], LABEL[a])
        _band(axes[1], steps, [[r["old_loss"] - rr["log"][0]["old_loss"] for r in rr["log"]] for rr in rs], C[a], LABEL[a])
        cv = F["arms"][a]["curve_mean"]
        axes[2].plot([cv["new_loss"][0] - v for v in cv["new_loss"]], cv["forgetting"], color=C[a], marker="o", ms=2.5,
                     label=LABEL[a])
    axes[0].set_ylabel("new-text loss, nats per character"); axes[0].set_title("What it learns", fontsize=10)
    axes[1].set_ylabel("rise of the old-text loss (forgetting)"); axes[1].set_title("What it forgets", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel("step on the new text (32 windows of 128 characters)")
    axes[2].axvline(F["arms"]["full"]["curve_mean"]["new_loss"][0] - F["matched"]["new_loss"], color=MUTED, lw=0.8, ls=":")
    axes[2].set_xlabel("new-text gain (loss at step 0 minus now)"); axes[2].set_ylabel("forgetting")
    axes[2].set_title("The trade, every 25 steps (mean of 3 seeds)\ndotted: the matched gain", fontsize=10)
    axes[1].legend(fontsize=8.5, loc="upper left")
    _save(fig, "forgetting.png")


def fig_one_number():
    F = read_results("forget")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, ax, key in ((0, axes[0], "kl_new"), (1, axes[1], "kl_old")):
        for a in ARMS:
            for rr in _runs(a):
                lg = rr["log"][1:]
                ax.scatter([r[key] for r in lg], [r["old_loss"] - rr["log"][0]["old_loss"] for r in lg], s=6,
                           color=C[a], alpha=0.35, lw=0)
                ax.scatter([rr["final"][key]], [rr["final"]["forgetting"]], s=34, color=C[a], edgecolor="white",
                           lw=0.6, label=LABEL[a] if rr["seed"] == 0 else None, zorder=3)
        on = F["one_number"]
        rho = on[f"spearman_final_{key}"]
        where = "on the new text" if key == "kl_new" else "on the old text"
        ax.set_xlabel(f"KL(start || model) {where}, nats per character")
        ax.set_title(f"KL {where}: Spearman {rho:.2f} over the 18 final points", fontsize=10)
    axes[0].set_ylabel("forgetting (rise of the old-text loss)")
    axes[0].legend(fontsize=8.5)
    _save(fig, "one_number.png")


def fig_plasticity():
    P = read_results("plasticity")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axes[0]
    for a in PLAST:
        y = np.array(P["arms"][a]["online_acc_mean_curve"])
        w = 20 if a != "fresh" else 1
        ys = np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y
        if a == "fresh":
            ax.axhline(100 * y.mean(), color=C[a], ls="--", lw=1.2, label=f"{LABEL[a]} (mean of {len(y)} tasks)")
        else:
            ax.plot(np.arange(len(ys)) + w, 100 * ys, color=C[a], label=LABEL[a])
    ax.set_xlabel("task (FashionMNIST, pixels shuffled anew; 6,000 images, one pass)")
    ax.set_ylabel("online accuracy, % (3 seeds, 20-task mean)")
    ax.set_title("Does each new task go as well as the first?", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    for ax, k, lab in ((axes[1], "dead", "hidden units never active, %"), (axes[2], "erank", "effective rank, last hidden layer")):
        for a in ("backprop", "cbp", "sgd"):
            h = P["arms"][a]["health"][k]
            ax.plot(h["tasks"], [100 * v if k == "dead" else v for v in h["mean"]], color=C[a], label=LABEL[a])
        ax.set_xlabel("task"); ax.set_ylabel(lab)
        ax.set_title("Dead units" if k == "dead" else "Variety of the features", fontsize=10)
    _save(fig, "plasticity.png")


def fig_library():
    L = read_results("library")
    M = read_results("library_memory")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for a in LIB:
        lg = L["arms"][a]["log"]
        axes[0].plot([r["step"] for r in lg], [r["new_nll"] for r in lg], color=C[a], marker="o", ms=3, label=LABEL[a])
        axes[1].plot([r["kl_new"] for r in lg], [r["probe_nll"] - lg[0]["probe_nll"] for r in lg], color=C[a],
                     marker="o", ms=4, label=LABEL[a])
    axes[0].set_xlabel("step (4 GSM8K chats each)"); axes[0].set_ylabel("GSM8K test loss, nats per token")
    axes[0].set_title("SmolLM2-360M-Instruct learns the new task", fontsize=10)
    axes[0].legend(fontsize=8.5)
    axes[1].set_xlabel("KL(start || model) on GSM8K replies, nats per token")
    axes[1].set_ylabel("rise of the 500-item probe loss")
    axes[1].set_title(f"Forgetting against KL: Spearman {L['one_number']['spearman_kl_new']:.2f} (16 points)", fontsize=10)
    ax = axes[2]
    gb = 2**30
    parts = [("base weights (bf16)", M["base_weight_bytes"]), ("LoRA weights", M["lora_r8_weight_bytes"]),
             ("LoRA gradients", M["runs"]["plain"]["lora_grad_bytes"]), ("Adam state", M["runs"]["plain"]["adam_state_bytes"]),
             ("saved for backward", M["runs"]["plain"]["saved_for_backward_bytes"]),
             ("saved, checkpointing", M["runs"]["checkpointing"]["saved_for_backward_bytes"])]
    cols = [MUTED, C["lora8"], C["lora8"], C["lora8"], C["backprop"], C["cbp"]]
    ax.barh([p for p, _ in parts][::-1], [v / gb for _, v in parts][::-1], color=cols[::-1])
    for i, (_, v) in enumerate(parts[::-1]):
        ax.text(v / gb + 0.01, i, f"{v / 2**20:,.0f} MB", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("GB on the GPU")
    ax.set_title(f"One LoRA r8 step, {M['batch']} chats of {M['tokens_per_sequence']} tokens", fontsize=10)
    ax.set_xlim(0, max(v for _, v in parts) / gb * 1.35)
    _save(fig, "library.png")


def main() -> None:
    for name, fn in (("start", fig_start), ("forget", fig_forgetting), ("forget", fig_one_number),
                     ("plasticity", fig_plasticity), ("library", fig_library)):
        if (RESULTS / f"{name}.json").exists():
            fn()


if __name__ == "__main__":
    main()
