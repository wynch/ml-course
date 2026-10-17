"""Redraw every figure of module 16 from results/*.json; no model is loaded.

    uv run python src/figures.py
"""

from __future__ import annotations

import math

import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ORDER = ("random", "mae", "jepa", "simclr", "supervised")
LABEL = {"random": "random encoder (floor)", "mae": "masked reconstruction", "jepa": "I-JEPA-lite",
         "simclr": "SimCLR", "supervised": "supervised (ceiling)"}
COL = {"random": "#8a8f8d", "mae": "#e0a100", "jepa": "#1d6fb8", "simclr": "#d1495b", "supervised": "#3fa34d"}
VCOL = {"no stop-grad": "#d1495b", "stop-grad": "#e0a100", "EMA": "#1d6fb8", "VICReg var": "#7b4fb0", "SIGReg": "#3fa34d"}
VARIANTS = ("no-stop-grad", "stop-grad", "ema", "vicreg-var", "sigreg")


def pct(v: float) -> str:
    """Percent with one decimal, halves rounded up (0.4735 -> 47.4%), as in the README."""
    return f"{math.floor(v * 1000 + 0.5 + 1e-9) / 10:.1f}%"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)


def pretrain_all() -> dict:
    return {o: common.read_results(f"pretrain_{o}") for o in ORDER}


def fig_probes(P: dict, dino: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = [LABEL[o] for o in ORDER] + ["DINOv2-small (22M params, 142M images)"]
    vals = [P[o]["probes"]["class_acc"] for o in ORDER] + [dino["class_acc"]]
    cols = [COL[o] for o in ORDER] + ["#bfc5c3"]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    ax.barh(y[-1], vals[-1], color="none", edgecolor="#5d6b67", hatch="//", height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, pct(v), va="center", fontsize=9)
    ax.axvline(P["random"]["probes"]["class_acc"], color="#8a8f8d", ls=":", lw=1)
    ax.axvline(P["supervised"]["probes"]["class_acc"], color="#3fa34d", ls=":", lw=1)
    ax.set_yticks(y, names, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("CIFAR-10 test accuracy of a linear probe on frozen features")
    b = P["simclr"]["budget_seconds"]
    ax.set_title(f"One {P['simclr']['encoder_params']:,}-parameter CNN, {b:.0f} s of training per objective; "
                 f"DINOv2 is a reference", fontsize=9.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "probes.png", dpi=150)
    plt.close(fig)


def fig_collapse(C: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for r in C:
        lg = r["log"]
        t = [p["seconds"] for p in lg]
        c = VCOL[r["variant"]]
        axes[0].plot(t, [p["std_norm"] for p in lg], color=c, lw=1.8, label=r["variant"])
        axes[1].plot(t, [p["erank"] for p in lg], color=c, lw=1.8)
        tt = [p["seconds"] for p in lg if p["pred_loss"] is not None]
        axes[2].plot(tt, [p["pred_loss"] for p in lg if p["pred_loss"] is not None], color=c, lw=1.8)
    axes[0].set_ylabel("std of L2-normalised latents × √128")
    axes[0].set_title("Latent spread (0 = all point one way)", fontsize=10)
    axes[1].set_ylabel("effective rank (of 128)")
    axes[1].set_title("Directions in use", fontsize=10)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("prediction loss (MSE on target cells)")
    axes[2].set_title("The loss the encoder minimises", fontsize=10)
    for ax in axes:
        ax.set_xlabel("training seconds")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8.5, frameon=False)
    fig.suptitle(f"The same JEPA objective, five ways, {C[0]['budget_seconds']:.0f} s each on CIFAR-10", fontsize=11)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "collapse.png", dpi=150)
    plt.close(fig)


def fig_keeps(P: dict, dino: dict) -> None:
    keys = (("class_acc", "class (10 classes; chance 10%)"), ("colour_r2", "mean colour (R²)"),
            ("rotation_acc", "rotation (4 angles; chance 25%)"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    names = [LABEL[o].split(" (")[0] for o in ORDER] + ["DINOv2-small"]
    for ax, (k, title) in zip(axes, keys):
        vals = [P[o]["probes"][k] for o in ORDER] + [dino[k]]
        cols = [COL[o] for o in ORDER] + ["#bfc5c3"]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=cols, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(max(v, 0) + 0.01, yi, f"{v:.2f}" if k == "colour_r2" else f"{v:.0%}", va="center", fontsize=8.5)
        ax.set_yticks(y, names if ax is axes[0] else [""] * len(names), fontsize=9)
        ax.set_xlim(0, 1.12)
        ax.set_title(title, fontsize=10)
        _style(ax)
    fig.suptitle("What a linear probe can read from each frozen encoder (CIFAR-10 test set)", fontsize=11)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "keeps.png", dpi=150)
    plt.close(fig)


def fig_views(P: dict) -> None:
    s, m, j = P["simclr"]["sample"], P["mae"]["sample"], P["jepa"]["sample"]
    panels = [("original", s["original"], None), ("SimCLR view 1", s["view1"], None), ("SimCLR view 2", s["view2"], None),
              ("masked reconstruction:\nencoder input (60% hidden)", m["masked"], None),
              ("I-JEPA-lite: context\n(outlined: target cells)", j["context"], j["target_mask"])]
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.8))
    for ax, (title, img, tm) in zip(axes, panels):
        ax.imshow(np.asarray(img, dtype=np.uint8), interpolation="nearest")
        if tm is not None:
            for r in range(8):
                for c in range(8):
                    if tm[r][c]:
                        ax.add_patch(Rectangle((c * 4 - 0.5, r * 4 - 0.5), 4, 4, fill=False, ec="#ffd400", lw=1.1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("What each objective sees for one test image (32 × 32 pixels; the grid cells are 4 × 4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "views.png", dpi=150)
    plt.close(fig)


def main() -> None:
    common.FIGURES.mkdir(exist_ok=True)
    P = pretrain_all()
    dino = common.read_results("dinov2")
    C = [common.read_results(f"collapse_{v}") for v in VARIANTS]
    fig_probes(P, dino)
    fig_collapse(C)
    fig_keeps(P, dino)
    fig_views(P)
    print("wrote figures/probes.png, collapse.png, keeps.png, views.png")


if __name__ == "__main__":
    main()
