"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

Each script also calls its own function here when run with --figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, read_results  # noqa: E402

ACCENT = "#9a6a00"      # track 8
GREY = "#7d8b87"
INK = "#1c2422"
OTHER = "#453781"


def _save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def distil() -> None:
    r = read_results("distil")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    for run in r["runs"]:
        c = ACCENT if run["kind"] == "distil" else GREY
        steps = [p["step"] for p in run["curve"]][1:]
        a.plot(steps, [p["val_loss"] for p in run["curve"]][1:], color=c, lw=1.4, alpha=0.9,
               label=("distilled from the teacher" if run["kind"] == "distil" else "from scratch")
               if run["seed"] == r["setup"]["seeds"][0] else None)
    a.axhline(r["teacher"]["val_loss"], color=INK, ls="--", lw=1, label=f"teacher ({r['teacher']['params']:,} params)")
    a.set_xlabel("training step (batch 64 × 128 characters)")
    a.set_ylabel("validation loss (nats per character)")
    a.set_title(f"Student: {r['student']['params']:,} params, three seeds each", fontsize=10)
    a.legend(fontsize=9, frameon=False)
    for s in r["setup"]["seeds"]:
        sc = next(x["val_loss"] for x in r["runs"] if x["kind"] == "scratch" and x["seed"] == s)
        di = next(x["val_loss"] for x in r["runs"] if x["kind"] == "distil" and x["seed"] == s)
        b.plot([0, 1], [sc, di], color=GREY, lw=1)
        b.scatter([0], [sc], color=GREY, zorder=3)
        b.scatter([1], [di], color=ACCENT, zorder=3)
        b.text(1.06, di, f"seed {s}", fontsize=8, va="center")
    b.set_xticks([0, 1], ["from scratch", "distilled"])
    b.set_xlim(-0.3, 1.5)
    b.set_title(f"Final loss, step {r['setup']['steps']}, paired by seed", fontsize=10)
    _save(fig, "distil.png")


def prune() -> None:
    r = read_results("prune")
    base = r["rows"][0]["ppl"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for method, c in (("magnitude", ACCENT), ("random", GREY)):
        rows = [x for x in r["rows"] if x["method"] == method]
        xs = [0] + [100 * x["fraction"] for x in rows]
        ys = [base] + [x["ppl"] for x in rows]
        ax.plot(xs, ys, "o-", color=c, label=f"{method} (smallest |w| first)" if method == "magnitude"
                else "random positions (control)")
        for x, y in zip(xs[1:], ys[1:]):
            ax.annotate(f"{y:,.1f}" if y < 1e4 else f"{y:.1e}", (x, y), textcoords="offset points", xytext=(6, -2),
                        fontsize=8)
    ax.axhline(base, color=INK, ls="--", lw=1, label=f"untouched, {base:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("weights zeroed in each block matrix (%)")
    ax.set_ylabel("perplexity on the wikitext slice (log)")
    ax.set_xticks([0, 10, 30, 50])
    ax.set_title(f"SmolLM2-360M-Instruct, {r['ppl_setup']['predicted_tokens']:,} tokens", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    _save(fig, "prune.png")


def quant() -> None:
    import numpy as np

    r = read_results("quant")
    rows = r["rows"]
    sm = r["shown_matrix"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9))
    for i, x in enumerate(rows):
        c = INK if i == 0 else ACCENT
        a.scatter(x["bytes"] / 1e6, x["ppl"], color=c, s=40, zorder=3)
        name = "untouched (bf16)" if i == 0 else x["scheme"]
        a.annotate(f"{name}\n{x['ppl']:.2f}", (x["bytes"] / 1e6, x["ppl"]), textcoords="offset points",
                   xytext=(6, 4), fontsize=8)
    a.set_xlabel("bytes to store the weights (MB)")
    a.set_ylabel("perplexity on the wikitext slice")
    a.set_xlim(0, max(x["bytes"] for x in rows) / 1e6 * 1.35)
    a.set_ylim(top=max(x["ppl"] for x in rows) * 1.12)
    a.set_title("Size against damage", fontsize=10)
    edges = np.array(sm["hist_edges"])
    mids = (edges[:-1] + edges[1:]) / 2
    for (name, h), c in zip(sm["hist"].items(), (OTHER, GREY, ACCENT)):
        b.step(mids, np.maximum(h, 0.5), where="mid", color=c,
               label=f"{name}: rel. error {sm['rel_error'][name]:.3f}")
    b.set_yscale("log")
    b.set_ylim(0.8, None)
    b.set_xlabel("quantised − original weight (clipped to ±0.25)")
    b.set_ylabel("weights (log)")
    b.set_title(f"Errors in one matrix: {sm['name'].replace('model.', '').replace('.weight', '')}", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    _save(fig, "quant.png")


def mlx_way() -> None:
    r = read_results("mlx_way")
    q = read_results("quant")
    labels = ["torch, float32\nuntouched", "torch, fake\n4-bit g64", "MLX bf16\nuntouched", "MLX 4-bit g64\n(mlx_lm.convert)"]
    ppl = [q["rows"][0]["ppl"], q["rows"][3]["ppl"], r["rows"][0]["ppl"], r["rows"][1]["ppl"]]
    tps = [q["rows"][0]["speed"]["median_tok_s"], q["rows"][3]["speed"]["median_tok_s"],
           r["rows"][0]["speed"]["median_tok_s"], r["rows"][1]["speed"]["median_tok_s"]]
    colors = [GREY, GREY, OTHER, ACCENT]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, vals, title, fmt in ((a, ppl, "Perplexity, same tokens", "{:.2f}"),
                                 (b, tps, "Tokens per second, 128 greedy tokens", "{:.0f}")):
        ax.bar(range(4), vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_title(title, fontsize=10)
    _save(fig, "mlx_way.png")


def main() -> None:
    distil()
    prune()
    quant()
    mlx_way()


if __name__ == "__main__":
    main()
