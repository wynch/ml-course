"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

Each script also redraws its own figure with --figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, RESULTS, read_results  # noqa: E402

COLORS = {"attention": "#453781", "linear": "#7d8590", "decay": "#b7410e",
          "delta": "#1f918d", "gated_delta": "#d99a3e"}
NAMES = {"attention": "softmax attention", "linear": "linear attention", "decay": "decay (module 18's form)",
         "delta": "delta rule", "gated_delta": "gated delta rule"}
CAP_STYLE = {
    "linear": ("linear", "-"), "decay, a = 0.95": ("decay", "-"), "delta": ("delta", "-"),
    "gated delta, a = 0.95": ("gated_delta", "-"), "delta, 10 passes": ("delta", "--"),
}


def _save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def fig_capacity() -> None:
    r = read_results("capacity")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, key, title in ((axes[0], "curves", "Random keys"),
                           (axes[1], "curves_correlated", f"Correlated keys (cosine near {r['rho']})")):
        for lab, c in r[key].items():
            rule, ls = CAP_STYLE[lab]
            ax.plot(r["ns"], c["recall"], ls, color=COLORS[rule], marker="o", ms=3, label=lab)
        ax.axvline(r["d_k"], color="#999", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ticks = [1, 4, 16, 32, 64, 128]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlabel(f"pairs written into one {r['d_k']} × {r['d_k']} state")
        ax.set_ylabel("share of keys read back right")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    ax = axes[2]
    n = r["by_age_n"]
    for lab, ys in r["by_age"].items():
        rule, ls = CAP_STYLE[lab]
        ax.plot(range(1, n + 1), ys, ls, color=COLORS[rule], lw=1.4, label=lab)
    ax.set_xlabel(f"pair number in write order (1 = oldest)")
    ax.set_ylabel("share read back right")
    ax.set_title(f"Which pairs survive: {n} random pairs", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    _save(fig, "capacity.png")


def fig_forms() -> None:
    r = read_results("forms")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for dev, ls in (("cpu", "-"), ("mps", "--")):
        if dev not in r["ms_forward_backward_gated_delta"]:
            continue
        t = r["ms_forward_backward_gated_delta"][dev]
        xs = [int(x) for x in t]
        ax.plot(xs, [t[str(x)]["recurrent"] for x in xs], ls, color="#7d8590", marker="o", ms=3, label=f"loop, {dev.upper()}")
        ax.plot(xs, [t[str(x)]["chunked"] for x in xs], ls, color=COLORS["gated_delta"], marker="o", ms=3,
                label=f"chunks of {r['chunk']}, {dev.upper()}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(r["lengths"], [f"{x:,}" for x in r["lengths"]])
    ax.set_yscale("log")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("ms, forward + backward")
    ax.set_title(f"Gated delta rule, batch {r['batch']}, {r['heads']} heads of {r['d_k']}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "forms.png")


def fig_mqar() -> None:
    r = read_results("mqar")
    pairs = r["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    for key, g in sorted(r["grid"].items()):
        if g["d_head"] != 16:
            continue
        ax.plot(pairs, [g["mean"][str(n)] for n in pairs], color=COLORS[g["rule"]], marker="o", ms=3.5,
                label=f"{NAMES[g['rule']]}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(pairs, [str(p) for p in pairs])
    ax.set_xlabel("key-value pairs in the sequence")
    ax.set_ylabel("query accuracy (mean of seeds)")
    ax.set_title("Two layers, heads of 16, width 64", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[1]
    n = str(pairs[-1])
    for rule in ("linear", "decay", "delta", "gated_delta", "attention"):
        pts = sorted((g["state_bytes"], g["mean"][n], g["spread"][n]) for g in r["grid"].values() if g["rule"] == rule)
        if not pts:
            continue
        xs, ys, sp = zip(*pts)
        ax.errorbar(xs, ys, yerr=[s / 2 for s in sp], color=COLORS[rule], marker="o" if rule != "attention" else "s",
                    ms=4, capsize=2, label=NAMES[rule] + (f" (KV cache at T = {r['T']})" if rule == "attention" else ""))
    for g in r["grid"].values():
        if g["learned_seeds"] == 0:
            ax.annotate("never left chance", (g["state_bytes"], g["mean"][n]), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, color=COLORS[g["rule"]])
    for c in r.get("check", {}).values():
        ax.plot([4 * 2 * r["d_model"] * c["d_head"]], [c["mean"][n]], "o", mfc="none", color=COLORS[c["rule"]], ms=7,
                label="gated delta, gate starting at 0.99 (check)")
    ax.set_xscale("log", base=2)
    xt = sorted({g["state_bytes"] for g in r["grid"].values()})
    ax.set_xticks(xt, [f"{x:,}" for x in xt])
    ax.set_xlabel("bytes kept between tokens (float32, both layers)")
    ax.set_ylabel(f"query accuracy at {n} pairs")
    ax.set_title("Recall against the size of the state", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "mqar.png")


def fig_hybrid() -> None:
    r = read_results("mqar")
    pairs = r["pairs"]
    order = ["all_gated_delta", "attention_at_1", "attention_at_3", "all_attention"]
    labels = ["4 × gated delta", "attention 2nd of 4", "attention 4th of 4", "4 × attention"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    w = 0.8 / len(pairs)
    shades = ["#d9e7e6", "#a9cfcc", "#6fb3ae", "#37918c", "#1b5f5c"][: len(pairs)]
    for j, n in enumerate(pairs):
        ys = [r["hybrid"][k]["mean"][str(n)] if k in r["hybrid"] else 0 for k in order]
        ax.bar([i + (j - 2) * w for i in range(len(order))], ys, w, color=shades[j], label=f"{n} pairs")
    ax.set_xticks(range(len(order)), labels, fontsize=8)
    for i, k in enumerate(order):
        h = r["hybrid"].get(k)
        if h and h["learned_seeds"] < len(h["seeds"]):
            miss = len(h["seeds"]) - h["learned_seeds"]
            ax.text(i, max(h["mean"].values()) + 0.03, f"{miss} of {len(h['seeds'])} seeds never left chance",
                    ha="center", fontsize=7.5, color="#5d6b67")
    ax.set_ylabel("query accuracy (mean of seeds)")
    ax.set_title("Four layers, heads of 8: where the one attention layer goes", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "hybrid.png")


def fig_library() -> None:
    r = read_results("library")
    names = {"qwen3": "Qwen3-0.6B (28 attention layers)", "qwen3.5": "Qwen3.5-0.8B (18 gated delta + 6 attention)"}
    colors = {"qwen3": COLORS["attention"], "qwen3.5": COLORS["gated_delta"]}
    xs = r["contexts"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for k, p in r["models"].items():
        c = p["contexts"]
        axes[0].plot(xs, [c[str(x)]["cache_bytes"] / 2**20 for x in xs], color=colors[k], marker="o", label=names[k])
        axes[1].plot(xs, [c[str(x)]["prefill_s_median"] for x in xs], color=colors[k], marker="o", label=names[k])
        axes[2].plot(xs, [c[str(x)]["decode_tokens_per_s_median"] for x in xs], color=colors[k], marker="o", label=names[k])
    for ax, yl, t in ((axes[0], "MB held after the context", "Cache after the context"),
                      (axes[1], "seconds to read the context", "Prefill"),
                      (axes[2], "new tokens per second", "Decode")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [f"{x:,}" for x in xs])
        ax.set_xlabel("context, tokens")
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_yscale("log")
    axes[2].set_ylim(bottom=0)
    _save(fig, "library.png")


def main() -> None:
    for name, fn in (("capacity", fig_capacity), ("forms", fig_forms), ("mqar", fig_mqar),
                     ("mqar", fig_hybrid), ("library", fig_library)):
        if (RESULTS / f"{name}.json").exists():
            fn()


if __name__ == "__main__":
    main()
