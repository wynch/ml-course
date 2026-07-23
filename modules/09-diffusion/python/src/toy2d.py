"""Diffusion in 2D — the whole idea on a point cloud.

Small enough to train in seconds and to *see* every part of the process:
the forward noising, the reverse denoising, and the learned vector field.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .embeddings import SinusoidalTimeEmbedding


# --------------------------------------------------------------------------- #
# 2D datasets (normalised to roughly unit scale so the schedules behave)
# --------------------------------------------------------------------------- #
def make_spiral(n: int = 2000, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0.15, 1.0, n)) * 3.0 * np.pi
    r = t / (3.0 * np.pi)
    x = r * np.cos(t)
    y = r * np.sin(t)
    pts = np.stack([x, y], axis=1) + rng.normal(0, noise, (n, 2))
    return _normalize(pts).astype(np.float32)


def make_moons(n: int = 2000, noise: float = 0.06, seed: int = 0):
    """Two interleaving half-moons. Returns (points, labels)."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    t0 = np.pi * rng.uniform(0, 1, n0)
    t1 = np.pi * rng.uniform(0, 1, n1)
    m0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    m1 = np.stack([1 - np.cos(t1), 0.5 - np.sin(t1)], axis=1)
    pts = np.concatenate([m0, m1], axis=0) + rng.normal(0, noise, (n, 2))
    labels = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.int64)
    pts = _normalize(pts).astype(np.float32)
    return pts, labels


def _normalize(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0, keepdims=True)
    pts = pts / (pts.std(axis=0, keepdims=True) + 1e-8)
    return pts


# --------------------------------------------------------------------------- #
# The noise-prediction network: a small MLP with a time (and optional class)
# embedding. Given (x_t, t) it predicts the noise eps that was added.
# --------------------------------------------------------------------------- #
class DenoiseMLP(nn.Module):
    def __init__(self, data_dim: int = 2, hidden: int = 128, time_dim: int = 64,
                 num_classes: int | None = None):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.num_classes = num_classes
        if num_classes is not None:
            # +1 row for the "null" / unconditional token used by CFG.
            self.class_embed = nn.Embedding(num_classes + 1, time_dim)

        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor | None = None) -> torch.Tensor:
        cond = self.time_mlp(self.time_embed(t))
        if self.num_classes is not None:
            if y is None:  # unconditional -> null token index == num_classes
                y = torch.full((x.shape[0],), self.num_classes,
                               device=x.device, dtype=torch.long)
            cond = cond + self.class_embed(y)
        h = torch.cat([x, cond], dim=-1)
        return self.net(h)


# --------------------------------------------------------------------------- #
# Training: the DDPM objective is just MSE between true and predicted noise.
# --------------------------------------------------------------------------- #
def train_toy(model, diffusion, data, *, epochs=2000, batch=256, lr=2e-3,
              device="cpu", labels=None, cfg_drop=0.0, log_every=200, seed=0):
    torch.manual_seed(seed)
    model.to(device).train()
    data = torch.as_tensor(data, device=device)
    if labels is not None:
        labels = torch.as_tensor(labels, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = data.shape[0]
    losses = []
    for epoch in range(epochs):
        idx = torch.randint(0, n, (batch,), device=device)
        x0 = data[idx]
        t = torch.randint(0, diffusion.timesteps, (batch,), device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)

        kw = {}
        if labels is not None:
            y = labels[idx].clone()
            if cfg_drop > 0:  # randomly drop the label to train the null token
                drop = torch.rand(batch, device=device) < cfg_drop
                y[drop] = model.num_classes
            kw["y"] = y

        pred = model(x_t, t, **kw)
        loss = ((pred - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:5d}/{epochs} | loss {loss.item():.4f}")
    model.eval()
    return losses


# --------------------------------------------------------------------------- #
# The learned denoising vector field: at a grid of points, -eps points back
# toward the data manifold (it is proportional to the score of the noised data).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def vector_field(model, diffusion, t: int, device="cpu", lim=2.5, grid=18,
                 y: int | None = None):
    xs = torch.linspace(-lim, lim, grid)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(device)
    t_b = torch.full((pts.shape[0],), t, device=device, dtype=torch.long)
    kw = {}
    if y is not None:
        kw["y"] = torch.full((pts.shape[0],), y, device=device, dtype=torch.long)
    eps = model(pts, t_b, **kw)
    vec = -eps  # denoising direction
    return (gx.numpy(), gy.numpy(),
            vec[:, 0].reshape(grid, grid).cpu().numpy(),
            vec[:, 1].reshape(grid, grid).cpu().numpy())


@torch.no_grad()
def sample_with_cfg(model, diffusion, n, device, y: int, guidance: float,
                    record_every: int | None = None):
    """Classifier-free-guidance sampling in 2D.

    eps = eps_uncond + guidance * (eps_cond - eps_uncond).
    guidance=0 -> unconditional, 1 -> plain conditional, >1 -> amplified.
    """
    x = torch.randn(n, 2, device=device)
    y_c = torch.full((n,), y, device=device, dtype=torch.long)
    y_n = torch.full((n,), model.num_classes, device=device, dtype=torch.long)
    traj = [x.clone()] if record_every else None
    for t in reversed(range(diffusion.timesteps)):
        t_b = torch.full((n,), t, device=device, dtype=torch.long)
        eps_c = model(x, t_b, y=y_c)
        eps_u = model(x, t_b, y=y_n)
        eps = eps_u + guidance * (eps_c - eps_u)

        beta_t = diffusion.betas[t]
        sqrt_om = diffusion.sqrt_one_minus_alphas_cumprod[t]
        mean = (x - beta_t / sqrt_om * eps) / torch.sqrt(diffusion.alphas[t])
        if t == 0:
            x = mean
        else:
            x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
        if record_every and (t % record_every == 0 or t == 0):
            traj.append(x.clone())
    return x, traj
