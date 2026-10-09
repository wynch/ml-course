"""Signature lab: diffusion on a 2D spiral. Trains in ~1 minute on MPS/CPU.

Produces four artefacts in ../figures/:
  forward_process.png   forward noising panel (data dissolving into Gaussian)
  reverse_process.gif   pure noise organising back into the spiral
  vector_field.png      learned denoising field (quiver) at three timesteps
  toy2d_loss.png        training loss
"""

from __future__ import annotations

import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from src.embeddings import get_device
from src.schedules import Diffusion
from src.toy2d import DenoiseMLP, make_spiral, train_toy, vector_field

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
SCHEDULE = "cosine"
TIMESTEPS = 200
LIM = 2.8


def plot_forward(data, diffusion, device):
    ts = [0, 20, 50, 100, 150, 199]
    fig, axes = plt.subplots(1, len(ts), figsize=(15, 2.7))
    x0 = torch.as_tensor(data, device=device)
    for ax, t in zip(axes, ts):
        tb = torch.full((x0.shape[0],), t, device=device, dtype=torch.long)
        xt = diffusion.q_sample(x0, tb).cpu().numpy()
        ax.scatter(xt[:, 0], xt[:, 1], s=3, c="#3b6ea5", alpha=0.5)
        ax.set_title(f"t = {t}", fontsize=11)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Forward process: structure dissolves into Gaussian noise",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "forward_process.png", dpi=90)
    plt.close(fig)


def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(losses, lw=0.8, c="#3b6ea5")
    # smoothed overlay
    k = 25
    sm = np.convolve(losses, np.ones(k) / k, mode="valid")
    ax.plot(range(k - 1, len(losses)), sm, lw=2, c="#b5452f", label="smoothed")
    ax.set_xlabel("training step"); ax.set_ylabel("noise-MSE loss")
    ax.set_title("2D diffusion training loss"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "toy2d_loss.png", dpi=90); plt.close(fig)


def plot_vector_field(model, diffusion, device):
    ts = [180, 100, 20]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, t in zip(axes, ts):
        gx, gy, vx, vy = vector_field(model, diffusion, t, device=device,
                                      lim=LIM, grid=17)
        mag = np.sqrt(vx ** 2 + vy ** 2)
        ax.quiver(gx, gy, vx, vy, mag, cmap="viridis", scale=28, width=0.004)
        ax.set_title(f"denoising field at t = {t}", fontsize=11)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Learned score / denoising direction (−ε predicted by the MLP)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "vector_field.png", dpi=72); plt.close(fig)


def make_reverse_gif(model, diffusion, device, n=1500):
    _, traj = diffusion.p_sample_loop(
        model, (n, 2), device, record_every=4)
    frames = [f.cpu().numpy() for f in traj]
    total = diffusion.timesteps

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=4, c="#b5452f", alpha=0.6)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    title = ax.set_title("")

    def update(i):
        scat.set_offsets(frames[i])
        t_left = max(total - i * 4, 0)
        title.set_text(f"reverse step — t ≈ {t_left}")
        return scat, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    anim.save(FIG / "reverse_process.gif", writer=PillowWriter(fps=18), dpi=70)
    plt.close(fig)


def main():
    device = get_device()
    print(f"device: {device}")
    data = make_spiral(n=2000, seed=0)
    diffusion = Diffusion.from_schedule(SCHEDULE, TIMESTEPS, device=device)
    model = DenoiseMLP(hidden=128, time_dim=64)

    t0 = time.time()
    losses = train_toy(model, diffusion, data, epochs=2500, batch=256,
                       lr=2e-3, device=device, log_every=500)
    print(f"trained in {time.time() - t0:.1f}s | final loss {losses[-1]:.4f}")

    plot_forward(data, diffusion, device)
    plot_loss(losses)
    plot_vector_field(model, diffusion, device)
    make_reverse_gif(model, diffusion, device)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
