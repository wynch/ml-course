"""Redraw every figure of module 30 from results/*.json; nothing is trained.

    uv run python src/figures.py
"""

from __future__ import annotations

import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RULES = ("backprop", "checkpointing", "mezo", "forward_forward", "mono_forward", "predictive_coding")
LABEL = {"backprop": "backprop", "checkpointing": "backprop + checkpointing", "mezo": "MeZO (zeroth-order)",
         "forward_forward": "forward-forward", "mono_forward": "Mono-Forward", "predictive_coding": "predictive coding",
         "inference": "inference only", "lora": "LoRA, backprop", "base": "untrained",
         "backprop_sgd": "backprop, plain SGD", "mono_forward_sgd": "Mono-Forward, plain SGD"}
COL = {"backprop": "#1c2422", "checkpointing": "#8a8f8d", "mezo": "#d1495b", "forward_forward": "#e0a100",
       "mono_forward": "#7a5cc2", "predictive_coding": "#3fa34d", "lora": "#1c2422", "inference": "#b9c2bf",
       "backprop_sgd": "#1c2422", "mono_forward_sgd": "#7a5cc2"}
DASH = {"checkpointing": (0, (4, 3)), "backprop_sgd": ":", "mono_forward_sgd": ":"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def _save(fig, name):
    common.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(common.FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{name}.png")


def race() -> None:
    T = common.read_results("train")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axs[0]
    for r in RULES:
        c = T["rules"][r]["mean_curve"]
        ax.plot(range(1, len(c) + 1), [100 * v for v in c], color=COL[r], lw=2 if r != "checkpointing" else 1.4,
                ls=DASH.get(r, "-"), label=f"{LABEL[r]}: {100 * T['rules'][r]['final_mean']:.1f}%")
    fu = T["followup"]
    mean = [sum(c[e] for c in fu["curves"]) / len(fu["curves"]) for e in range(fu["epochs"])]
    ax.plot(range(1, len(mean) + 1), [100 * v for v in mean], color=COL["mezo"], lw=1.4, ls=":",
            label=f"MeZO at half the rate, 40 epochs (follow-up): {100 * fu['final_mean']:.1f}%")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"])
    ax.set_xlabel("epochs (log scale); MeZO's NaN epochs fall off the bottom")
    ax.set_ylabel("test accuracy, %, mean of 3 seeds")
    ax.set_ylim(40, 92)
    ax.set_title("Accuracy per epoch, FashionMNIST")
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)
    ax = axs[1]
    xs = [T["rules"][r]["seconds_per_epoch_mean"] for r in RULES]
    ys = [100 * T["rules"][r]["final_mean"] for r in RULES]
    for r, x, y in zip(RULES, xs, ys):
        ax.scatter([x], [y], s=70, color=COL[r], zorder=3)
        off = {"checkpointing": (6, -12), "backprop": (-10, 8), "forward_forward": (6, -12), "mono_forward": (-10, 8)}
        ax.annotate(LABEL[r], (x, y), textcoords="offset points", xytext=off.get(r, (6, 4)), fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("seconds per epoch, CPU, 4 threads, shared machine (log scale)")
    ax.set_ylabel("test accuracy after 20 epochs, %")
    ax.set_title("What each epoch costs in time")
    _style(ax)
    _save(fig, "race")


def memory() -> None:
    M = common.read_results("memory")
    depths = M["depths"]
    rules = ("predictive_coding", "backprop", "checkpointing", "forward_forward", "mono_forward",
             "backprop_sgd", "mono_forward_sgd", "mezo")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axs[0]
    for r in rules:
        ys = [M["by_depth"][str(d)]["rules"][r]["extra_mb"] for d in depths]
        ax.plot(depths, ys, marker="o", color=COL[r], ls=DASH.get(r, "-"), label=LABEL[r])
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths, [str(d) for d in depths])
    ax.set_xlabel("hidden layers of 1,024 (batch 2,048)")
    ax.set_ylabel("memory above inference, MiB (live tensors)")
    ax.text(depths[0], 0, "  MeZO: 0 at every depth", fontsize=8, color=COL["mezo"], va="bottom")
    ax.set_title("What training adds to inference")
    ax.legend(fontsize=8)
    _style(ax)
    ax = axs[1]
    d16 = M["by_depth"][str(depths[-1])]
    live = [d16["rules"][r]["extra_mb"] for r in rules]
    rss = [d16["rules"][r]["rss_extra_mb"] for r in rules]
    y = np.arange(len(rules))
    ax.barh(y - 0.2, live, 0.38, color=[COL[r] for r in rules], label="live tensors")
    ax.barh(y + 0.2, rss, 0.38, color="none", edgecolor=[COL[r] for r in rules], hatch="///",
            label="process (resident memory)")
    ax.set_yticks(y, [LABEL[r] for r in rules], fontsize=8)
    ax.invert_yaxis()
    adam = 2 * d16["params_mb"]
    ax.axvline(adam, color="#5c6b67", lw=0.8, ls="--")
    ax.text(adam, len(rules) - 0.4, f" Adam's two buffers: {adam:.0f}", fontsize=7.5, color="#5c6b67")
    ax.set_xlabel(f"memory above inference at {depths[-1]} hidden layers, MiB")
    ax.set_title(f"At {depths[-1]} hidden layers ({d16['params_mb']:.0f} MiB of weights)")
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)
    _save(fig, "memory")


def credit() -> None:
    C = common.read_results("credit")
    net = C["nets"]["8 hidden"]
    L = len(net["cosine"])
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    t = np.arange(C["steps"] + 1)
    for l in range(L):
        col = cmap(l / (L - 1))
        lab = "lowest layer" if l == 0 else ("output layer" if l == L - 1 else None)
        cos = [np.nan if v is None else v for v in net["cosine"][l]]
        axs[0].plot(t[1:], cos[1:], color=col, lw=1.4, label=lab)
        size = [v if v > 0 else np.nan for v in net["size_ratio"][l]]
        axs[1].plot(t[1:], size[1:], color=col, lw=1.4, label=lab)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("inference steps T (log scale)")
    axs[0].set_ylabel("cosine with backprop's gradient")
    axs[0].set_title("Direction: right on arrival, then drifts")
    axs[0].legend(fontsize=8)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("inference steps T (log scale)")
    axs[1].set_ylabel("size of update / size of backprop's gradient")
    axs[1].set_title("Size: shrinks about tenfold per layer down")
    axs[1].legend(fontsize=8)
    for ax in axs:
        _style(ax)
    fig.suptitle("Predictive coding, 8 hidden layers of 256, at initialisation, 256 test images", fontsize=10)
    _save(fig, "credit")


def library() -> None:
    Lb = common.read_results("library", common.RESULTS / "library")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    for arm, col in (("lora", COL["lora"]), ("mezo", COL["mezo"])):
        e = Lb[arm]["evals"]
        ax.plot([max(r["step"], 1) for r in e], [100 * r["val_acc"] for r in e], marker="o", color=col,
                label=f"{'LoRA, backprop' if arm == 'lora' else 'MeZO, every weight'}: {100 * Lb[arm]['final_val_acc']:.1f}%")
    ax.axhline(100 * Lb["base_val_acc"], color="#8a8f8d", ls="--", lw=1)
    ax.text(1.1, 100 * Lb["base_val_acc"] + 0.6, f"untrained {100 * Lb['base_val_acc']:.1f}%", fontsize=8, color="#5c6b67")
    ax.set_xscale("log")
    ax.set_xlabel("training steps, batch 16 (log scale; step 0 drawn at 1)")
    ax.set_ylabel("SST-2 validation accuracy, %")
    ax.set_title("SmolLM2-360M-Instruct, 872 validation sentences")
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)
    ax = axs[1]
    names = ["inference", "mezo", "lora"]
    driver = [Lb["base_memory"]["mps_driver_peak_gb"], Lb["mezo"]["mps_driver_peak_gb"], Lb["lora"]["mps_driver_peak_gb"]]
    ax.bar(range(3), driver, color=[COL["inference"], COL["mezo"], COL["lora"]])
    for i, v in enumerate(driver):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), ["inference\n(evaluation)", "MeZO", "LoRA\nbackprop"])
    ax.set_ylabel("peak memory held by the Metal driver, GB")
    ax.set_title("Memory on the GPU")
    _style(ax)
    _save(fig, "library")


def main() -> None:
    for f in (race, memory, credit, library):
        try:
            f()
        except FileNotFoundError as e:
            print(f"  skipped {f.__name__}: {e}")


if __name__ == "__main__":
    main()
