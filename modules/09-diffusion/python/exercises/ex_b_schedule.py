"""Exercise (b) — noise schedules: linear vs cosine, compared by reverse GIF.

The schedule is the sequence of beta_t (how much noise is injected per step).
The *original* DDPM used a linear schedule; improved DDPM introduced a cosine
schedule that adds noise more gently at the ends, so more of the timesteps do
useful work. You will implement the cosine schedule, then this script trains the
2D spiral model under each schedule and renders both reverse-process GIFs plus
a plot of their alpha_bar curves so you can compare.

    uv run python exercises/ex_b_schedule.py

Outputs: figures/ex_b_linear.gif, ex_b_cosine.gif, ex_b_schedules.png
"""

from __future__ import annotations

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from src.embeddings import get_device
from src.schedules import Diffusion, linear_beta_schedule
from src.toy2d import DenoiseMLP, make_spiral, train_toy

FIG = pathlib.Path(__file__).resolve().parents[1].parent / "figures"


def my_cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Return the (timesteps,) tensor of betas for the cosine schedule.

    Recipe (Nichol & Dhariwal 2021):
      1. define f(t) = cos( ((t/T) + s)/(1+s) * pi/2 )**2  for t = 0..T
      2. alpha_bar_t = f(t) / f(0)
      3. beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
      4. clip betas into [1e-8, 0.999]
    """
    # TODO(you): implement the cosine schedule and return betas of shape (timesteps,)
    raise NotImplementedError("implement my_cosine_beta_schedule")


def build_diffusion(schedule: str, timesteps: int, device):
    if schedule == "linear":
        betas = linear_beta_schedule(timesteps).to(device)
    else:
        betas = my_cosine_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, 0)
    return Diffusion(betas, alphas, acp, torch.sqrt(acp),
                     torch.sqrt(1 - acp), timesteps)


def reverse_gif(model, diffusion, device, path, lim=2.8, n=1200):
    _, traj = diffusion.p_sample_loop(model, (n, 2), device, record_every=4)
    frames = [f.cpu().numpy() for f in traj]
    fig, ax = plt.subplots(figsize=(4, 4))
    scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=4, c="#b5452f", alpha=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    anim = FuncAnimation(fig, lambda i: scat.set_offsets(frames[i]),
                         frames=len(frames), interval=60)
    anim.save(path, writer=PillowWriter(fps=18), dpi=68)
    plt.close(fig)


def main():
    device = get_device()
    data = make_spiral(1500, seed=0)
    T = 200
    diffs = {}
    for sched in ["linear", "cosine"]:
        print(f"training under {sched} schedule...")
        diffusion = build_diffusion(sched, T, device)
        diffs[sched] = diffusion
        model = DenoiseMLP()
        train_toy(model, diffusion, data, epochs=2000, device=device, log_every=0)
        reverse_gif(model, diffusion, device, FIG / f"ex_b_{sched}.gif")
        print(f"  wrote ex_b_{sched}.gif")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for sched, c in [("linear", "#3b6ea5"), ("cosine", "#b5452f")]:
        ax.plot(diffs[sched].alphas_cumprod.cpu(), label=sched, lw=2, c=c)
    ax.set_xlabel("timestep t"); ax.set_ylabel(r"$\bar\alpha_t$ (signal retained)")
    ax.set_title("Linear vs cosine: how fast the signal decays"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "ex_b_schedules.png", dpi=90); plt.close(fig)
    print("wrote ex_b_schedules.png")


if __name__ == "__main__":
    main()
