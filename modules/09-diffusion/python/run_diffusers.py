"""The Hugging Face way: the same idea, production-grade, in a few lines.

Loads a pretrained DDPM (google/ddpm-cifar10-32) with `diffusers`, generates
samples, then swaps the DDPM sampler for DDIM and shows the quality-vs-steps
trade-off with wall-clock timings.

Produces in ../figures/:
  diffusers_ddpm_samples.png   samples from the pretrained pipeline
  ddim_steps_quality.png       DDIM at 10 / 50 / 200 steps + timings
"""

from __future__ import annotations

import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMPipeline, DDIMScheduler

from src.embeddings import get_device

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
MODEL = "google/ddpm-cifar10-32"


def grid_from_images(images, path, title, ncol=8):
    nrow = int(np.ceil(len(images) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow + 0.5))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        if i < len(images):
            ax.imshow(images[i])
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=85); plt.close(fig)


def main():
    device = get_device()
    print(f"device: {device}")
    pipe = DDPMPipeline.from_pretrained(MODEL).to(device)
    print(f"model: {MODEL} | UNet params: "
          f"{sum(p.numel() for p in pipe.unet.parameters()):,}")

    # 1) Generate from the stock DDPM pipeline (its own 1000-step scheduler,
    #    here shortened to 100 steps so it finishes quickly).
    gen = torch.Generator(device="cpu").manual_seed(0)
    out = pipe(batch_size=16, num_inference_steps=100, generator=gen)
    grid_from_images(out.images, FIG / "diffusers_ddpm_samples.png",
                     "Pretrained google/ddpm-cifar10-32 via DDPMPipeline (100 steps)")

    # 2) Swap the scheduler DDPM -> DDIM. Same UNet weights, deterministic,
    #    far fewer steps for similar quality. Show 10 / 50 / 200 steps.
    ddim = DDIMScheduler.from_config(pipe.scheduler.config)
    unet = pipe.unet
    step_counts = [10, 50, 200]
    n = 6
    rows, timings = [], []
    for steps in step_counts:
        ddim.set_timesteps(steps, device=device)
        g = torch.Generator(device="cpu").manual_seed(1)
        x = torch.randn(n, 3, 32, 32, generator=g).to(device)
        t0 = time.time()
        with torch.no_grad():
            for t in ddim.timesteps:
                eps = unet(x, t).sample
                x = ddim.step(eps, t, x).prev_sample
        dt = time.time() - t0
        timings.append(dt)
        imgs = ((x.clamp(-1, 1) + 1) / 2).cpu().permute(0, 2, 3, 1).numpy()
        rows.append(imgs)
        print(f"DDIM {steps:3d} steps | {n} imgs | {dt:.2f}s "
              f"({dt / steps * 1000:.0f} ms/step)")

    fig, axes = plt.subplots(len(step_counts), n, figsize=(n, len(step_counts) + 1))
    for r, (steps, imgs, dt) in enumerate(zip(step_counts, rows, timings)):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c]); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{steps} steps\n{dt:.1f}s", fontsize=9)
    fig.suptitle("DDIM: quality vs. number of inference steps (same UNet, same seed)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "ddim_steps_quality.png", dpi=85); plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
