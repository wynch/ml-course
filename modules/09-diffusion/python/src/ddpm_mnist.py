"""A tiny DDPM for 28x28 images, written from scratch.

A compact UNet (~a few hundred K params) with a sinusoidal time embedding,
trained with the same noise-prediction MSE objective as the 2D lab — only the
network changes. Kept deliberately small and readable so it trains in a
~10-20 minute budget on Apple MPS and you can read every layer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import SinusoidalTimeEmbedding


class ResBlock(nn.Module):
    """Two conv layers + GroupNorm/SiLU, with the time embedding added in."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28 UNet with skip connections."""

    def __init__(self, ch: int = 32, time_dim: int = 128, in_ch: int = 1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(in_ch, ch, 3, padding=1)
        # down
        self.d1 = ResBlock(ch, ch, time_dim)
        self.d2 = ResBlock(ch, ch * 2, time_dim)
        self.down = nn.AvgPool2d(2)
        # bottleneck
        self.mid = ResBlock(ch * 2, ch * 2, time_dim)
        # up
        self.u2 = ResBlock(ch * 2 + ch * 2, ch * 2, time_dim)
        self.u1 = ResBlock(ch * 2 + ch, ch, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1)
        )

    def forward(self, x, t):
        temb = self.time_mlp(t)
        x = self.in_conv(x)                 # 28, ch
        h1 = self.d1(x, temb)               # 28, ch
        h2 = self.d2(self.down(h1), temb)   # 14, 2ch
        m = self.mid(self.down(h2), temb)   # 7,  2ch
        u = F.interpolate(m, scale_factor=2, mode="nearest")     # 14
        u = self.u2(torch.cat([u, h2], dim=1), temb)             # 14, 2ch
        u = F.interpolate(u, scale_factor=2, mode="nearest")     # 28
        u = self.u1(torch.cat([u, h1], dim=1), temb)             # 28, ch
        return self.out(u)


def load_fashion(n: int, seed: int = 0):
    """Load a subsample of FashionMNIST from HF, scaled to [-1, 1], shape (n,1,28,28)."""
    from datasets import load_dataset

    ds = load_dataset("zalando-datasets/fashion_mnist", split="train")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    imgs = np.stack([np.array(ds[int(i)]["image"], dtype=np.float32) for i in idx])
    imgs = imgs / 127.5 - 1.0  # [-1, 1]
    return torch.from_numpy(imgs)[:, None, :, :]


def train_ddpm(model, diffusion, data, *, epochs=30, batch=128, lr=2e-4,
               device="cpu", log_every=1, snapshot_epochs=None, seed=0):
    """Train and optionally snapshot a fixed-noise sample grid at given epochs."""
    torch.manual_seed(seed)
    model.to(device).train()
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = data.shape[0]
    steps_per_epoch = max(n // batch, 1)
    losses, snapshots = [], {}
    fixed_noise = torch.randn(16, 1, 28, 28, device=device)
    snapshot_epochs = set(snapshot_epochs or [])

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * batch:(s + 1) * batch]
            x0 = data[idx]
            t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise)
            pred = model(x_t, t)
            loss = ((pred - noise) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        ep_loss /= steps_per_epoch
        losses.append(ep_loss)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}/{epochs} | loss {ep_loss:.4f}")
        if (epoch + 1) in snapshot_epochs:
            model.eval()
            grid, _ = diffusion.p_sample_loop(model, (16, 1, 28, 28), device)
            snapshots[epoch + 1] = grid.cpu()
            model.train()
    model.eval()
    return losses, snapshots, fixed_noise
