"""Redraw every figure from results/*.json; no model is loaded.

    uv run python src/figures.py
"""

from __future__ import annotations

from common import FIGURES, read_results

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"net": "#6e7a14", "least_squares": "#1c2422", "ridge_0.1": "#2a78d6", "knn_3": "#b8336a",
          "one_gd_step": "#eb6834", "control_net_on_shuffled": "#8a8f8d", "main_net_on_shuffled": "#b0a060"}
NAMES = {"net": "2-layer transformer", "least_squares": "least squares", "ridge_0.1": "ridge λ 0.1",
         "knn_3": "3 nearest neighbours", "one_gd_step": "one gradient step",
         "control_net_on_shuffled": "control net, shuffled y"}


def _save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png")


def fig_icl() -> None:
    r = read_results("icl")
    m = r["mse_over_d"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    steps, loss = zip(*r["train_loss_every_100"]["main"])
    ax[0].plot(steps, loss, color=COLORS["net"], lw=0.8, label="main net (one batch every 100 steps)")
    s2, l2 = zip(*r["train_loss_every_100"]["control"])
    ax[0].plot(s2, l2, color=COLORS["control_net_on_shuffled"], lw=0.8, label="control net, shuffled y")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("training step")
    ax[0].set_ylabel("training loss, MSE / d over all n")
    ax[0].set_title("Training")
    ax[0].legend(fontsize=8)
    for a, log in zip(ax[1:], (False, True)):
        for k in NAMES:
            v = np.array(m[k])
            if log:
                v = np.maximum(v, 1e-6)
            a.plot(m["n"], v, "-o" if k == "net" else "-", ms=3, lw=2 if k == "net" else 1.2,
                   color=COLORS[k], label=NAMES[k])
        a.axvline(5, color="#999", lw=0.6, ls=":")
        a.set_xlabel("pairs in the prompt (n)")
        a.set_ylabel("MSE / d (0 is perfect, 1 is guessing 0)")
        a.set_xticks(range(0, 21, 2))
    ax[1].set_ylim(0, 1.5)
    ax[1].set_title("Error by number of pairs")
    ax[2].set_yscale("log")
    ax[2].set_ylim(1e-4, 3)
    ax[2].set_title("Same, log scale (least squares is 0 from n = 5)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "icl")


def fig_gd() -> None:
    r = read_results("gd_demo")
    im = r["implied"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(im["n"], im["cos_net_ls"], "-o", ms=3, color=COLORS["least_squares"], label="net vs least squares")
    ax[0].plot(im["n"], im["cos_net_gd"], "-o", ms=3, color=COLORS["one_gd_step"], label="net vs one gradient step")
    ax[0].plot(im["n"], im["cos_gd_ls"], "--", color="#999", label="one gradient step vs least squares")
    ax[0].axvline(5, color="#999", lw=0.6, ls=":")
    ax[0].set_xlabel("pairs in the prompt (n)")
    ax[0].set_ylabel("mean cosine of weight vectors")
    ax[0].set_title("The net's implied weights")
    ax[0].legend(fontsize=8)
    layers = r["config"]["layers"]
    x = np.arange(len(layers))
    pr = r["probes_r2"]
    for i, (net, tgt, c, off) in enumerate((("trained", "least_squares", COLORS["least_squares"], -0.3),
                                             ("trained", "one_gd_step", COLORS["one_gd_step"], -0.1),
                                             ("untrained", "least_squares", "#bbb", 0.1),
                                             ("untrained", "one_gd_step", "#ddd", 0.3))):
        ax[1].bar(x + off, pr[net][tgt], 0.2, color=c, edgecolor="#555", lw=0.5,
                  label=f"{net} net → {NAMES[tgt]} w")
    ax[1].set_xticks(x, layers)
    ax[1].set_ylabel("probe R² on held-out prompts")
    ax[1].set_title(f"Linear probes at the query, n = {r['config']['probe_n']}")
    ax[1].axhline(0, color="#555", lw=0.6)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "gd_demo")


def fig_fewshot() -> None:
    r = read_results("fewshot")
    c = r["conditions"]
    names = list(c)
    acc = [c[k]["accuracy"] for k in names]
    lo = [c[k]["accuracy"] - c[k]["ci95"][0] for k in names]
    hi = [c[k]["ci95"][1] - c[k]["accuracy"] for k in names]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cols = [COLORS["net"]] * 4 + [COLORS["control_net_on_shuffled"]]
    ax.bar(range(len(names)), acc, color=cols[: len(names)], yerr=[lo, hi], capsize=4)
    ax.axhline(r["config"]["majority_share"], color="#555", ls="--", lw=0.8, label="always the majority label")
    ax.axhline(0.5, color="#999", ls=":", lw=0.8)
    for i, a in enumerate(acc):
        ax.text(i, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names)), [n.replace(", ", "\n") for n in names])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel(f"accuracy on {r['config']['n_test']} SST-2 sentences")
    ax.set_title("SmolLM2-360M-Instruct, few-shot (95% intervals)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "fewshot")


def fig_line() -> None:
    r = read_results("line")
    d = r["demo"]
    m = r["mse_over_d"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(d["u"], d["y"], color="#1c2422", zorder=3, label="the 5 points in the prompt")
    ax[0].plot(d["grid"], d["least_squares"], color=COLORS["least_squares"], label="least squares line")
    ax[0].plot(d["grid"], d["net"], "--", color=COLORS["net"], lw=2, label="net's guess at each query")
    ax[0].set_xlabel("u")
    ax[0].set_ylabel("y")
    ax[0].set_title("The explorable's net on one prompt")
    ax[0].legend(fontsize=8)
    ax[1].plot(m["n"], m["net"], "-o", ms=3, color=COLORS["net"], label="line net")
    ax[1].plot(m["n"], np.maximum(m["ridge_0.1"], 1e-6), color=COLORS["ridge_0.1"], label="ridge λ 0.1")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("points in the prompt (n)")
    ax[1].set_ylabel("MSE / d")
    ax[1].set_title("Its error on 1,000 held-out lines")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "line")


def main() -> None:
    for name, fn in (("icl", fig_icl), ("gd_demo", fig_gd), ("fewshot", fig_fewshot), ("line", fig_line)):
        try:
            fn()
        except FileNotFoundError:
            print(f"  skipped {name}: no results yet")


if __name__ == "__main__":
    main()
