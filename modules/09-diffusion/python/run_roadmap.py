"""Schematic figures for the 'road to Stable Diffusion' README section.

Pure matplotlib box-and-arrow diagrams (no model downloads). Produces:
  latent_diffusion.png   diffuse in a compressed VAE latent, not pixels
  text_conditioning.png  CLIP text embedding conditions the UNet via attention
  cfg.png                classifier-free guidance: extrapolate cond vs uncond
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, GREY = "#3b6ea5", "#b5452f", "#4a7c59", "#dfe4ea"


def box(ax, xy, w, h, text, color=GREY, fc=None, fontsize=10):
    fc = fc or color
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec="#333", lw=1.2, alpha=0.9))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def arrow(ax, p0, p1, text=None, color="#333"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=color))
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.06, text, ha="center", va="bottom", fontsize=8.5,
                color=color)


def fig_latent():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    box(ax, (0.02, 0.35), 0.12, 0.3, "image\n512×512×3", fc="#cfe0ee")
    box(ax, (0.20, 0.35), 0.10, 0.3, "VAE\nencoder", fc=GREEN)
    box(ax, (0.36, 0.30), 0.16, 0.4, "diffusion in\nLATENT space\n64×64×4", fc=BLUE)
    box(ax, (0.58, 0.35), 0.10, 0.3, "VAE\ndecoder", fc=GREEN)
    box(ax, (0.74, 0.35), 0.12, 0.3, "image\n512×512×3", fc="#cfe0ee")
    arrow(ax, (0.14, 0.5), (0.20, 0.5))
    arrow(ax, (0.30, 0.5), (0.36, 0.5), "compress ~48×")
    arrow(ax, (0.52, 0.5), (0.58, 0.5))
    arrow(ax, (0.68, 0.5), (0.74, 0.5), "reconstruct")
    ax.text(0.44, 0.14, "The UNet denoises a tiny latent, not megapixels — "
            "orders of magnitude cheaper.", ha="center", fontsize=9, style="italic")
    ax.set_xlim(0, 0.88); ax.set_ylim(0, 0.85); ax.axis("off")
    ax.set_title("Latent diffusion (Stable Diffusion): diffuse in a compressed latent",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "latent_diffusion.png", dpi=90); plt.close(fig)


def fig_text():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    box(ax, (0.02, 0.55), 0.24, 0.25, '"a cat astronaut"', fc="#f0e2c0")
    box(ax, (0.02, 0.18), 0.24, 0.25, "CLIP text\nencoder", fc=ORANGE)
    box(ax, (0.40, 0.30), 0.22, 0.4, "UNet\n(denoiser)", fc=BLUE)
    box(ax, (0.72, 0.35), 0.22, 0.3, "cross-attention\nlayers", fc=GREEN)
    arrow(ax, (0.14, 0.55), (0.14, 0.43), "tokenize")
    arrow(ax, (0.26, 0.30), (0.40, 0.42), "text embeddings")
    arrow(ax, (0.62, 0.5), (0.72, 0.5), "Q from image,\nK,V from text")
    ax.text(0.5, 0.06, "Text conditions the denoiser: image patches attend to word "
            "embeddings at every UNet block.", ha="center", fontsize=9, style="italic")
    ax.set_xlim(0, 0.96); ax.set_ylim(0, 0.9); ax.axis("off")
    ax.set_title("Text conditioning via CLIP + cross-attention", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "text_conditioning.png", dpi=90); plt.close(fig)


def fig_cfg():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    o = np.array([0.15, 0.2])
    uncond = np.array([0.55, 0.45])
    cond = np.array([0.6, 0.7])
    guided = uncond + 2.2 * (cond - uncond)
    for p, txt, col in [(uncond, "ε(uncond)", GREY),
                        (cond, "ε(cond)", BLUE),
                        (guided, "ε(guided)  =  ε_u + w·(ε_c − ε_u)", ORANGE)]:
        ax.add_patch(FancyArrowPatch(o, p, arrowstyle="-|>", mutation_scale=16,
                                     lw=2, color=col))
        ax.text(p[0] + 0.01, p[1] + 0.02, txt, fontsize=10, color=col)
    ax.add_patch(FancyArrowPatch(cond, guided, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.2, ls="--", color="#888"))
    ax.plot(*o, "ko"); ax.text(o[0] - 0.02, o[1] - 0.05, "x_t", fontsize=10)
    ax.text(0.5, 0.03, "Guidance scale w>1 pushes samples further toward the prompt "
            "(sharper, less diverse).", ha="center", fontsize=9, style="italic")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 0.95); ax.axis("off")
    ax.set_title("Classifier-free guidance: extrapolate away from unconditional",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "cfg.png", dpi=90); plt.close(fig)


def main():
    fig_latent(); fig_text(); fig_cfg()
    print("roadmap figures written to", FIG)


if __name__ == "__main__":
    main()
