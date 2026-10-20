"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

Each script also calls its own function here when run with --figures.
Arms keep one colour and one marker everywhere; the markers carry identity
for readers who cannot tell orange from amber.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, read_results  # noqa: E402

ACCENT = "#9a6a00"      # track 8; the ternary arm
INK = "#1c2422"
GREY = "#7d8b87"
ARM_STYLE = {           # colour, marker
    "ternary": (ACCENT, "o"),
    "4-bit": ("#2a78d6", "s"),
    "2-bit": ("#eb6834", "^"),
    "16-bit": ("#1baf7a", "D"),
}
KV_STYLE = {            # (keys layout, rotation) -> colour, marker, line style
    ("channel", False): ("#2a78d6", "s", "-"),
    ("channel", True): ("#2a78d6", "s", "--"),
    ("token", False): ("#eb6834", "^", "-"),
    ("token", True): ("#eb6834", "^", "--"),
}


def _save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def _plain_log(ax, ticks) -> None:
    """Log y axis with plain numbers at the given ticks."""
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())


def _label(arm: str) -> str:
    return {"ternary": "ternary, trained in", "4-bit": "float, rounded to 4 bits",
            "2-bit": "float, rounded to 2 bits", "16-bit": "float, stored in 16 bits"}[arm]


def matched() -> None:
    r = read_results("matched")
    f = read_results("followup") if (FIGURES.parent / "results" / "followup.json").exists() else None
    fig, axes = plt.subplots(1, 2 if f else 1, figsize=(11.5 if f else 7, 4.4), sharey=True, squeeze=False)
    panels = [("Pre-registered: rate tuned on the middle shape", {(c["budget_mb"], c["arm"]): c for c in r["cells"]},
               {(c["budget_mb"], c["arm"]): c["tried"] for c in r["cells"]})]
    if f:
        panels.append(("Follow-up: the one-block shape gets its own rate",
                       {(c["budget_mb"], c["arm"]): {**c["best"], "arm": c["arm"]} for c in f["cells"]},
                       {(c["budget_mb"], c["arm"]): r_tried + [{"bytes": c["shallow"]["bytes"], "report": x["report"]}
                                                              for x in c["shallow"]["rates"]]
                        for c, r_tried in zip(f["cells"], [x["tried"] for x in r["cells"]])}))
    for ax, (title, best, tried) in zip(axes[0], panels):
        for arm, (c, m) in ARM_STYLE.items():
            keys = sorted(k for k in best if k[1] == arm)
            for k in keys:
                ax.scatter([t["bytes"] / 1e6 for t in tried[k]], [t["report"] for t in tried[k]], color=c, marker=m,
                           s=18, alpha=0.3, lw=0)
            xs = [best[k]["bytes"] / 1e6 for k in keys]
            ys = [best[k]["report_mean"] for k in keys]
            es = [best[k]["report_sd"] or 0 for k in keys]
            ax.errorbar(xs, ys, yerr=es, color=c, marker=m, ms=8, lw=2, capsize=3, label=_label(arm))
        ax.set_xlabel("file size (MB): every byte counted")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(1.55, 2.45)
    axes[0][0].set_ylabel("held-out loss (nats per character)")
    axes[0][0].legend(fontsize=9, frameon=False, loc="upper right")
    fig.suptitle("Best shape and rate per arm, mean ± sd of 3 seeds; faint: every shape and rate tried (seed 0)",
                 fontsize=10, y=1.0)
    _save(fig, "matched.png")


def export() -> None:
    r = read_results("export")
    rows = r["rows"]
    lrs = [x["lr"] for x in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = range(len(rows))
    share = [100 * (x["share_reverted"] or 0) for x in rows]
    a.bar(list(xs), share, 0.6, color=ACCENT)
    for i, x in enumerate(rows):
        a.annotate(f"{x['changed_then_reverted_by_bf16']:,} of {x['codes_changed']:,}", (i, share[i]),
                   textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    a.set_xticks(list(xs), [f"{x:g}" for x in lrs])
    a.set_ylim(0, max(share) * 1.25)
    a.set_xlabel("fine-tuning rate (200 steps, AdamW)")
    a.set_ylabel("changed codes the bf16 export undoes (%)")
    a.set_title(f"Of the codes the fine-tune changed ({r['base']['codes']:,} in all)", fontsize=10)
    w = 0.38
    b.bar([i - w / 2 for i in xs], [x["gain_fp32"] for x in rows], w, color=ACCENT, label="fp32 export")
    b.bar([i + w / 2 for i in xs], [x["gain_bf16"] for x in rows], w, color="#2a78d6", hatch="//", label="bf16 export")
    b.axhline(0, color=INK, lw=0.8)
    b.set_xticks(list(xs), [f"{x['lr']:g}" for x in rows])
    b.set_xlabel("fine-tuning rate")
    b.set_ylabel("Shakespeare loss drop (nats; up is better)")
    b.set_title("What the fine-tune bought, by export", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    _save(fig, "export.png")


def kv_hand() -> None:
    r = read_results("kv_hand")
    base = r["rows"][0]["ppl"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [3, 2]})
    for (layout, rot), (c, m, ls) in KV_STYLE.items():
        rows = sorted((x for x in r["rows"] if x["keys"] == layout and x["rotation"] == rot), key=lambda x: -x["bits"])
        a.plot([x["bits"] for x in rows], [x["ppl"] for x in rows], ls, marker=m, color=c,
               label=f"keys per {layout}{', rotated' if rot else ''}")
        last = rows[-1]
        a.annotate(f"{last['ppl']:.0f}", (last["bits"], last["ppl"]), textcoords="offset points", xytext=(-22, 0),
                   fontsize=8, va="center")
    a.axhline(base, color=INK, lw=1, ls=":", label=f"untouched bf16 cache, {base:.2f}")
    _plain_log(a, [16, 20, 30, 50, 100, 200, 500])
    a.set_xticks([8, 4, 3, 2])
    a.invert_xaxis()
    a.set_xlabel("bits per cached key and value (values always per token)")
    a.set_ylabel("perplexity (log)")
    a.set_title(f"SmolLM2-360M, {r['ppl_tokens']:,} tokens, weights untouched", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    s = r["outliers"]["sample"]
    import numpy as np

    kb, ka = np.abs(np.array(s["keys_before"])).max(0), np.abs(np.array(s["keys_after"])).max(0)
    ch = range(len(kb))
    b.bar([i - 0.2 for i in ch], kb, 0.4, color="#eb6834", label="as cached")
    b.bar([i + 0.2 for i in ch], ka, 0.4, color="#2a78d6", hatch="//", label="after the rotation")
    b.set_xlabel("channel of one key head")
    b.set_ylabel(f"max |k| over {s['tokens']} tokens")
    b.set_title(f"Layer {s['layer']}, key head {s['head']}: outlier channels", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    _save(fig, "kv_hand.png")


def kv_mlx() -> None:
    r = read_results("kv_mlx")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    rows = r["rows"][1:]
    xs = range(len(rows))
    w = 0.38
    a.bar([i - w / 2 for i in xs], [x["hand_ppl"] for x in rows], w, color="#eb6834", label="by hand (section 3)")
    a.bar([i + w / 2 for i in xs], [x["ppl"] for x in rows], w, color="#2a78d6", hatch="//",
          label="mlx-lm QuantizedKVCache")
    for i, x in enumerate(rows):
        a.annotate(f"{x['ppl']:.1f}", (i + w / 2, x["ppl"]), textcoords="offset points", xytext=(0, 3), fontsize=8,
                   ha="center")
    a.axhline(r["rows"][0]["ppl"], color=INK, lw=1, ls=":", label=f"bf16 cache, {r['rows'][0]['ppl']:.2f}")
    _plain_log(a, [16, 20, 30, 50, 100, 200, 500])
    a.set_xticks(list(xs), [f"{x['bits']} bits" for x in rows])
    a.set_ylabel("perplexity (log)")
    a.set_title("Keys and values per token, groups of 64", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    ctx = [x["tokens"] for x in r["cache_bytes"]]
    for k, (c, m) in {"bf16": (INK, "o"), "8-bit": ("#1baf7a", "D"), "4-bit": ("#2a78d6", "s"),
                      "2-bit": ("#eb6834", "^")}.items():
        ys = [x[k] / 2**20 for x in r["cache_bytes"]]
        b.plot(ctx, ys, marker=m, color=c, label=k)
        b.annotate(f"{ys[-1]:.0f} MB", (ctx[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), fontsize=8,
                   va="center")
    b.set_xlabel("tokens in the cache")
    b.set_ylabel("cache size (MB, from nbytes)")
    b.set_xticks(ctx)
    b.set_xlim(0, ctx[-1] * 1.22)
    b.set_title("The cache grows with the context", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    _save(fig, "kv_mlx.png")


if __name__ == "__main__":
    import os

    from common import RESULTS

    for name, fn in (("matched", matched), ("export", export), ("kv_hand", kv_hand), ("kv_mlx", kv_mlx)):
        if os.path.exists(RESULTS / f"{name}.json"):
            fn()
