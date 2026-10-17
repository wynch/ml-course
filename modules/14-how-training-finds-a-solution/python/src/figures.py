"""Redraw every README figure from the committed results/*.json, no training needed.

    uv run python src/figures.py

To recompute the numbers too, run each script with --figures instead.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import FIGURES, RESULTS, read_results  # noqa: E402

# Fixed categorical order: each entity keeps its colour in every figure.
C = {"sgd": "#2a78d6", "momentum": "#eb6834", "adam": "#1baf7a",
     "small": "#2a78d6", "large": "#eb6834", "wd1": "#2a78d6", "wd0": "#eb6834"}
LABEL = {"sgd": "SGD", "momentum": "momentum 0.9", "adam": "Adam"}
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


def fig_optim_toy():
    import optim_toy as ot

    r = read_results("optim_toy")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.2), gridspec_kw={"height_ratios": [1.15, 1]})
    for col, (name, f) in enumerate((("bowl", ot.bowl), ("rosenbrock", ot.rosenbrock))):
        L = r["losses"][name]
        ax = axes[0, col]
        (x0, x1), (y0, y1) = L["box"]
        X, Y = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
        Z = np.vectorize(lambda a, b: f(np.array([a, b])))(X, Y)
        ax.contour(X, Y, np.log10(Z + 1e-4), levels=18, colors=MUTED, linewidths=0.5, alpha=0.6)
        for opt, o in L["optimisers"].items():
            if not o["best_path"]:
                continue
            p = np.array(o["best_path"])
            ax.plot(p[:, 0], p[:, 1], color=C[opt], lw=1.3,
                    label=f"{LABEL[opt]}, lr {o['best_lr']:.3g}: {o['best_steps']:,} steps")
        ax.plot(*L["start"], "o", color=INK, ms=6)
        ax.plot(*L["minimum"], "*", color=INK, ms=10)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.grid(False)
        ax.set_title(f"{'Tilted bowl (condition number 25)' if name == 'bowl' else 'Rosenbrock'}: "
                     f"each at its best rate", fontsize=10, loc="left")
        ax.legend(loc="upper right" if name == "bowl" else "lower right", fontsize=8.5)

        ax = axes[1, col]
        for opt, o in L["optimisers"].items():
            lrs = [s["lr"] for s in o["sweep"] if s["steps"]]
            st = [s["steps"] for s in o["sweep"] if s["steps"]]
            ax.plot(lrs, st, "-o", ms=3.5, color=C[opt], label=f"{LABEL[opt]} ({o['lrs_that_reach']} of {len(r['lrs'])} rates)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel(f"steps to loss < {r['target']:g}")
        ax.set_title("Steps to the target at each rate (missing point: never within 10,000)", fontsize=10, loc="left")
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    _save(fig, "optim_toy.png")


def fig_optim_mlp():
    r = read_results("optim_mlp")
    runs = [x for x in r["runs"] if x["seed"] == 0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    ax = axes[0]
    w = 0.13
    for i, opt in enumerate(("sgd", "momentum", "adam")):
        for j, sched in enumerate(("constant", "warmup_cosine")):
            accs = []
            for lr in r["lrs"]:
                x = next(q for q in runs if q["optimiser"] == opt and q["schedule"] == sched and q["lr"] == lr)
                accs.append(x["test_acc"] or 0)
            pos = np.arange(len(r["lrs"])) + (i * 2 + j - 2.5) * w
            ax.bar(pos, accs, width=w * 0.9, color=C[opt], alpha=1.0 if sched == "warmup_cosine" else 0.45,
                   label=f"{LABEL[opt]}, {'warmup+cosine' if sched == 'warmup_cosine' else 'constant'}")
    ax.set_xticks(range(len(r["lrs"])), [f"{v:g}" for v in r["lrs"]])
    ax.set_xlabel("base learning rate")
    ax.set_ylabel("test accuracy after 10 epochs")
    ax.set_ylim(0, 0.95)
    ax.set_title("All 24 runs (pale: constant rate)", fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    for k, sched in enumerate(("constant", "warmup_cosine")):
        ax = axes[1 + k]
        for opt in ("sgd", "momentum", "adam"):
            b = r["best"][f"{opt}/{sched}"]
            x = next(q for q in runs if q["optimiser"] == opt and q["schedule"] == sched and q["lr"] == b["lr"])
            ep = [c["epoch"] for c in x["curve"]]
            ax.plot(ep, [c["test_acc"] for c in x["curve"]], color=C[opt],
                    label=f"{LABEL[opt]}, lr {b['lr']:g}")
        ax.set_ylim(0.70, 0.90)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"Best rate of each, {'warmup + cosine' if sched == 'warmup_cosine' else 'constant rate'}",
                     fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "optim_mlp.png")


def fig_landscape():
    r = read_results("landscape")
    fig = plt.figure(figsize=(13, 4.3))
    ax = fig.add_subplot(1, 3, 1)
    for name in ("small", "large"):
        n = r["nets"][name]
        ax.plot(r["line"], n["line_train"], color=C[name], label=f"batch {n['batch']}, train")
        ax.plot(r["line"], n["line_test"], color=C[name], ls="--", lw=1.3, label=f"batch {n['batch']}, test")
    ax.set_yscale("log")
    ax.set_xlabel("step along a filter-normalised direction")
    ax.set_ylabel("loss (5,000 images)")
    ax.set_title("1-D slice through each trained net", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    g = np.array(r["grid"])
    allz = np.log10(np.array([r["nets"][k]["surface_train"] for k in ("small", "large")]))
    levels = np.linspace(allz.min(), min(allz.max(), 1.5), 16)
    for k, name in enumerate(("small", "large")):
        ax = fig.add_subplot(1, 3, 2 + k)
        n = r["nets"][name]
        cs = ax.contourf(g, g, np.log10(np.array(n["surface_train"])), levels=levels, cmap="Blues_r", extend="max")
        ax.plot(0, 0, "+", color=INK, ms=9)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(f"2-D slice, batch {n['batch']:,}\ntop Hessian eigenvalue {n['top_eig']:.1f}", fontsize=10, loc="left")
        ax.set_xlabel("direction 1")
        if k == 0:
            ax.set_ylabel("direction 2")
    cb = fig.colorbar(cs, ax=fig.axes[1:], shrink=0.85)
    cb.set_label("log10 train loss")
    _save(fig, "landscape.png")


def fig_grok():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for key, label in (("wd1", "weight decay 1.0"), ("wd0", "weight decay 0")):
        if not (RESULTS / f"grok_{key}.json").exists():
            continue
        g = read_results(f"grok_{key}")
        cv = g["curve"]
        st = [max(c["step"], 1) for c in cv]
        axes[0].plot(st, [c["train_acc"] for c in cv], color=C[key], ls="--", lw=1.3, label=f"{label}, train")
        axes[0].plot(st, [c["test_acc"] for c in cv], color=C[key], label=f"{label}, test")
        axes[1].plot(st, [c["train_loss"] for c in cv], color=C[key], ls="--", lw=1.3, label=f"{label}, train")
        axes[1].plot(st, [c["test_loss"] for c in cv], color=C[key], label=f"{label}, test")
        axes[2].plot(st, [c["weight_norm"] for c in cv], color=C[key], label=label)
        for s, mk in ((g["train_hits_100"], "o"), (g["test_crosses_95"], "*")):
            if s:
                axes[0].plot(max(s, 1), 1.0 if mk == "o" else 0.95, mk, color=C[key], ms=8 if mk == "*" else 5)
    axes[0].axhline(0.95, color=MUTED, lw=0.8, ls=":")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy (dot: train 100%; star: test 95%)", fontsize=10, loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("cross-entropy")
    axes[1].set_title("Loss (exact zeros not drawn)", fontsize=10, loc="left")
    axes[2].set_ylabel("weight norm, all parameters")
    axes[2].set_title("Weight norm", fontsize=10, loc="left")
    axes[1].set_ylim(1e-6, 300)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("step (step 0 drawn at 1)")
    axes[2].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8.5)
    _save(fig, "grok.png")


def main() -> None:
    for fn, needs in ((fig_optim_toy, "optim_toy"), (fig_optim_mlp, "optim_mlp"),
                      (fig_landscape, "landscape"), (fig_grok, "grok_wd1")):
        if (RESULTS / f"{needs}.json").exists():
            fn()


if __name__ == "__main__":
    main()
