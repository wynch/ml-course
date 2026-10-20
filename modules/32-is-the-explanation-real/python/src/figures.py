"""Every figure of module 32, drawn from results/*.json alone (no model is loaded).

    uv run python src/figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, read_results

# Categorical order, fixed: trained, random, third series (dense / published), fourth.
C_TRAINED, C_RANDOM, C_THIRD, C_FOURTH = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9.5, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def fig_testbed() -> None:
    S, D = read_results("train_sparse"), read_results("train_dense")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    for R, c, lab in ((S, C_TRAINED, "sparse MLP: 2,048 neurons, 8 active"), (D, C_THIRD, "dense GELU MLP: 512 neurons")):
        st, lo = np.array(R["curve"]).T
        ax.plot(st, lo, color=c, lw=2, label=f"{lab} (held-out {R['val']['loss']:.3f})")
    for s, n in S["resampled"]:
        ax.axvline(s, color=GRID, lw=0.8, zorder=0)
    ax.set_ylim(1.2, 4.5)
    ax.set_xlabel("step (64 windows of 128 tokens)")
    ax.set_ylabel("training loss (nats per token)")
    ax.set_title("Two models, same data and steps")
    ax.legend(fontsize=8.5, loc="upper right")
    ax = axes[1]
    st = [s for s, _ in S["resampled"]]
    ax.bar(np.arange(len(st)), [n for _, n in S["resampled"]], color=C_TRAINED, width=0.6)
    ax.set_xticks(np.arange(len(st)), [f"{s:,}" for s in st])
    ax.set_xlabel("step")
    ax.set_ylabel("neurons resampled (both layers)")
    al = S["alive_per_layer_1e-5"]
    ax.set_title(f"Dead neurons get a new key; at the end {al[0]:,} and {al[1]:,} of 2,048 alive")
    fig.tight_layout()
    save(fig, "testbed.png")


def fig_truth(R: dict | None = None) -> None:
    R = R or read_results("dicts")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    edges = np.linspace(0, 1, 21)
    for ax, which in zip(axes[:2], ("trained", "random")):
        for kind, c, ls in (("transcoder", C_TRAINED if which == "trained" else C_RANDOM, "-"),
                            ("sae", C_TRAINED if which == "trained" else C_RANDOM, "--")):
            row = R["models"][which]["dicts"][kind]["s0"]
            h = np.array(row["best_cos_hist"]) / sum(row["best_cos_hist"])
            ax.step(edges, np.r_[h, h[-1]], where="post", color=c, ls=ls, lw=2,
                    label=f"{'transcoder' if kind == 'transcoder' else 'SAE'}: MMCS {row['mmcs_value']:.2f}")
        ax.axvline(0.9, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("best cosine with any decoder column")
        ax.set_ylabel("share of alive true neurons")
        ax.set_title(f"{which} model, seed 0 ({R['models'][which]['true_alive']:,} alive neurons)")
        ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[2]
    x = 0
    ticks, labs = [], []
    for which, c in (("trained", C_TRAINED), ("random", C_RANDOM)):
        for kind in ("sae", "transcoder"):
            K = R["models"][which]["dicts"][kind]
            vals = [K[f"s{s}"]["mmcs_value"] for s in (0, 1, 2)]
            ax.bar(x, np.mean(vals), color=c, width=0.7, alpha=1 if kind == "transcoder" else 0.55)
            ax.scatter([x] * 3, vals, color=INK, s=10, zorder=3)
            ax.text(x, np.mean(vals) + 0.02, f"{np.mean(vals):.2f}", ha="center", fontsize=8.5, color=INK)
            ticks.append(x); labs.append(f"{which}\n{'SAE' if kind == 'sae' else 'transcoder'}")
            x += 1
        x += 0.5
    lt = R["models"]["trained"].get("long_training")
    if lt:
        for xi, kind in ((0, "sae"), (1, "transcoder")):
            ax.scatter([xi + 0.25], [lt[kind]["mmcs_value"]], marker="D", s=26, color=C_FOURTH, zorder=4,
                       label="seed 0 at 10,000 steps (added later)" if kind == "sae" else None)
        ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xticks(ticks, labs, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("MMCS (dots: seeds 0, 1, 2)")
    ax.set_title("Mean best cosine to the truth")
    fig.tight_layout()
    save(fig, "truth.png")


def _hist_line(ax, hist, color, label, ls="-"):
    edges = np.linspace(0, 1, 21)
    h = np.array(hist) / max(1, sum(hist))
    ax.step(edges, np.r_[h, h[-1]], where="post", color=color, lw=2, ls=ls, label=label)


def fig_controls(R: dict | None = None) -> None:
    R = R or read_results("dicts")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, metric, title in ((axes[0], "detection", "Detection score: a 3-token rule predicts the unit"),
                              (axes[1], "token_r2", "Token R²: the current token alone explains it")):
        for which, c in (("trained", C_TRAINED), ("random", C_RANDOM)):
            s = R["models"][which]["dicts"]["sae"]["s0"][metric]
            _hist_line(ax, s["hist"], c, f"SAE latents, {which} model (median {s['median']:.2f})")
            t = R["models"][which]["true_neurons"][metric]
            _hist_line(ax, t["hist"], c, f"true neurons, {which} (median {t['median']:.2f})", ls=":")
        d = R["dense_neurons"][metric]
        _hist_line(ax, d["hist"], C_THIRD, f"dense model neurons (median {d['median']:.2f})", ls="--")
        ax.set_xlabel(metric.replace("_", " ").replace("r2", "R²"))
        ax.set_ylabel("share of scored units")
        ax.set_title(title)
        ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[2]
    x, ticks, labs = 0, [], []
    for which, c in (("trained", C_TRAINED), ("random", C_RANDOM)):
        for kind in ("sae", "transcoder"):
            S = R["models"][which]["dicts"][kind]["seeds"]
            for part, alpha in (("on_truth", 1.0), ("off_truth", 0.45)):
                v = S[part]["shared_share"]
                if v is not None:
                    ax.bar(x, v, color=c, alpha=alpha, width=0.8)
                    ax.text(x, v + 0.02, f"{v:.2f}\nn={S[part]['n']}", ha="center", fontsize=7, color=INK)
                ticks.append(x); labs.append(f"{'on' if part == 'on_truth' else 'off'}")
                x += 1
            ax.text(x - 1.5, -0.2, f"{which}\n{'SAE' if kind == 'sae' else 'transc.'}", ha="center", fontsize=8,
                    color=MUTED, transform=ax.transData)
            x += 0.6
    ax.set_xticks(ticks, labs, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share also found by seed 1")
    ax.set_title("Stable across seeds? on = matches a true neuron (≥ 0.9),\noff = matches none (< 0.5)",
                 fontsize=9.5)
    fig.tight_layout()
    save(fig, "controls.png")


def fig_attribution(A: dict | None = None) -> None:
    A = A or read_results("attribution")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, which, c in ((axes[0], "trained", C_TRAINED), (axes[1], "random", C_RANDOM)):
        M = A["models"][which]
        a, e = np.array(M["scatter"]["att"]), np.array(M["scatter"]["eff"])
        ax.scatter(a, e, s=8, color=c, alpha=0.5, linewidths=0)
        lim = max(np.abs(a).max(), np.abs(e).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("attribution (frozen model): activation × gradient")
        ax.set_ylabel("real drop in target logit when zeroed")
        ax.set_title(f"{which} model: r = {M['pearson_r_pooled']:.2f} over {len(a):,} active neurons")
    ax = axes[2]
    ks = ["0", "5", "10", "20", "40", "80"]
    for which, c in (("trained", C_TRAINED), ("random", C_RANDOM)):
        f = A["models"][which]["faithfulness_median"]
        ax.plot([int(k) for k in ks], [f[k] for k in ks], "o-", color=c, lw=2, ms=6, label=f"{which} model")
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xticks([0, 5, 10, 20, 40, 80], ["0", "5", "10", "20", "40", "80"])
    ax.set_xlabel("neurons kept (highest |attribution|); all other active neurons zeroed")
    ax.set_ylabel("P(target) kept / P(target) full, median")
    ax.set_title("Faithfulness of the top of the graph")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "attribution.png")
    fig_graph(A)


def fig_graph(A: dict) -> None:
    G = A["models"]["trained"]["graph"]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")
    ax.grid(False)
    nodes = G["nodes"]
    pos = {}
    for layer, y in ((0, 0.25), (1, 0.62)):
        row = sorted([n for n in nodes if n["layer"] == layer], key=lambda n: (n["pos"], n["neuron"]))
        for j, n in enumerate(row):
            pos[(layer, n["pos"], n["neuron"])] = ((j + 0.5) / max(1, len(row)), y)
    tgt = (0.5, 0.93)
    wmax = max([abs(e["w"]) for e in G["edges"]] + [1e-9])
    for e in G["edges"]:
        if abs(e["w"]) < 0.05 * wmax:
            continue
        (x0, y0), (x1, y1) = pos[tuple(e["src"])], pos[tuple(e["dst"])]
        ax.plot([x0, x1], [y0, y1], color=C_TRAINED if e["w"] > 0 else C_RANDOM,
                lw=0.5 + 4 * abs(e["w"]) / wmax, alpha=0.7, zorder=1)
    amax = max(abs(n["att"]) for n in nodes)
    for n in nodes:
        x, y = pos[(n["layer"], n["pos"], n["neuron"])]
        ax.plot([x, tgt[0]], [y, tgt[1]], color=GRID, lw=0.6, zorder=0)
        ax.scatter([x], [y], s=120 + 900 * abs(n["att"]) / amax, color=C_TRAINED if n["att"] > 0 else C_RANDOM,
                   edgecolor="white", linewidth=2, zorder=3)
        ex = " ".join(repr(t)[1:-1] for t in n["expl"][:3])
        ax.text(x, y - 0.075, f"L{n['layer']} #{n['neuron']} at {n['token']!r}\nfires on: {ex}\nattr {n['att']:+.2f}, "
                f"real {n['eff']:+.2f}", ha="center", va="top", fontsize=7, color=INK)
    ax.scatter([tgt[0]], [tgt[1]], s=300, color=INK, zorder=3)
    prompt = "".join(G["prompt"]).replace("\n", " ")
    ax.text(tgt[0], tgt[1] + 0.04, f"target: {G['target']!r} (p = {G['p_full']:.2f})", ha="center", fontsize=9,
            color=INK, weight="bold")
    ax.text(0, 1.13, "…" + prompt[-110:], fontsize=8, color=MUTED, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Prompt 0: the 6 neurons of highest |attribution| in each layer. Blue helps the target, "
                 "orange hurts; edge width = frozen-model edge weight", fontsize=9.5, loc="left", pad=22)
    save(fig, "graph.png")


def fig_library() -> None:
    L = read_results("library")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, metric in zip(axes, ("detection", "token_r2")):
        for name, c, lab in (("trained", C_TRAINED, "trained GPT-2, our SAE"), ("random", C_RANDOM, "random GPT-2 (init), our SAE"),
                             ("matched", C_FOURTH, "random GPT-2 (matched std), our SAE"),
                             ("published", C_THIRD, "trained GPT-2, published SAE")):
            s = L["saes"][name][metric]
            _hist_line(ax, s["hist"], c, f"{lab} (median {s['median']:.2f})", ls="--" if name == "published" else "-")
        ax.set_xlabel("detection score" if metric == "detection" else "token R²")
        ax.set_ylabel("share of scored latents")
        ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_title("GPT-2 small, residual before block 8: detection")
    axes[1].set_title("…and how much the current token alone explains")
    fig.tight_layout()
    save(fig, "library.png")


def main() -> None:
    fig_testbed()
    fig_truth()
    fig_controls()
    fig_attribution()
    fig_library()


if __name__ == "__main__":
    main()
