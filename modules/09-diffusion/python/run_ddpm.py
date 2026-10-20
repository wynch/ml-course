"""Mini-DDPM on FashionMNIST, from scratch. About 3 min on an Apple M5 (MPS).

Produces in ../figures/:
  ddpm_loss.png            training loss per epoch
  ddpm_dreaming.png        sample grids at training checkpoints (dreaming better)
  ddpm_trajectory.png      one sample's denoising trajectory, t=T -> 0
  ddpm_samples.png         final 8x8 sample grid
and ../results/ddpm.json (the numbers the README quotes).

The schedule is linear, rescaled by 1000 / T. The plain range 1e-4 -> 0.02 was
tuned for T = 1000; at T = 300 it leaves sqrt(alpha_bar_T) = 0.22, so sampling
from pure N(0, I) would start somewhere training never went. The cosine
schedule failed another way here (results/ddpm_cosine.json): its last betas
reach 0.999, this sampler does not clip the predicted image, and the samples
came out saturated black and white.
"""

from __future__ import annotations

import json
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.embeddings import get_device
from src.schedules import Diffusion
from src.ddpm_mnist import TinyUNet, load_fashion, train_ddpm

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
SCHEDULE = "scaled_linear"
TIMESTEPS = 300
N_IMAGES = 10000
EPOCHS = 40
SNAPSHOTS = [2, 10, 25, 40]


def to_img(x):
    return np.clip((x.squeeze().numpy() + 1) / 2, 0, 1)


def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", ms=3, c="#3b6ea5")
    ax.set_xlabel("epoch"); ax.set_ylabel("noise-MSE loss")
    ax.set_title("mini-DDPM training loss (FashionMNIST)")
    fig.tight_layout(); fig.savefig(FIG / "ddpm_loss.png", dpi=90); plt.close(fig)


def plot_dreaming(snapshots):
    epochs = sorted(snapshots)
    ncol = 8
    fig, axes = plt.subplots(len(epochs), ncol,
                             figsize=(ncol, len(epochs) + 0.6))
    for r, ep in enumerate(epochs):
        grid = snapshots[ep]
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(to_img(grid[c]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"epoch {ep}", fontsize=10)
    fig.suptitle("Watching the model dream better (same noise, more training)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "ddpm_dreaming.png", dpi=90); plt.close(fig)


def plot_trajectory(model, diffusion, device):
    torch.manual_seed(3)
    x = torch.randn(1, 1, 28, 28, device=device)
    snaps, tvals = [x.clone().cpu()], [diffusion.timesteps]
    keep = set(int(v) for v in np.linspace(diffusion.timesteps - 1, 0, 9))
    for t in reversed(range(diffusion.timesteps)):
        x = diffusion.p_sample_step(model, x, t)
        if t in keep:
            snaps.append(x.clone().cpu()); tvals.append(t)
    fig, axes = plt.subplots(1, len(snaps), figsize=(len(snaps) * 1.2, 1.6))
    for ax, img, tv in zip(axes, snaps, tvals):
        ax.imshow(to_img(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={tv}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Denoising trajectory for one sample (pure noise → image)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "ddpm_trajectory.png", dpi=90); plt.close(fig)


def plot_final_grid(model, diffusion, device):
    grid, _ = diffusion.p_sample_loop(model, (64, 1, 28, 28), device)
    grid = grid.cpu()
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(to_img(grid[i]), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Final samples from the from-scratch mini-DDPM", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "ddpm_samples.png", dpi=85); plt.close(fig)


def main():
    device = get_device()
    print(f"device: {device}")
    print(f"loading {N_IMAGES} FashionMNIST images...")
    data = load_fashion(N_IMAGES)
    diffusion = Diffusion.from_schedule(SCHEDULE, TIMESTEPS, device=device)
    model = TinyUNet(ch=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    t0 = time.time()
    losses, snapshots, _ = train_ddpm(
        model, diffusion, data, epochs=EPOCHS, batch=128, lr=2e-4,
        device=device, snapshot_epochs=SNAPSHOTS)
    train_min = (time.time() - t0) / 60
    print(f"trained in {train_min:.1f} min | final loss {losses[-1]:.4f}")

    plot_loss(losses)
    plot_dreaming(snapshots)
    plot_trajectory(model, diffusion, device)
    plot_final_grid(model, diffusion, device)
    total_min = (time.time() - t0) / 60
    lin = Diffusion.from_schedule("linear", TIMESTEPS)
    out = {
        "device": str(device), "schedule": SCHEDULE, "timesteps": TIMESTEPS,
        "n_images": N_IMAGES, "epochs": EPOCHS, "n_params": n_params,
        "train_minutes": round(train_min, 2),
        "train_and_figures_minutes": round(total_min, 2),
        "loss_first_epoch": losses[0], "loss_final": losses[-1], "losses": losses,
        "sqrt_alpha_bar_T": {SCHEDULE: diffusion.sqrt_alphas_cumprod[-1].item(),
                             "linear": lin.sqrt_alphas_cumprod[-1].item()},
    }
    (RESULTS / "ddpm.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"train + figures in {total_min:.1f} min")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
