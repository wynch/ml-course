"""Noise schedules and the forward (diffusion) process.

The heart of DDPM lives here. Two facts:

1. The forward process gradually turns data ``x0`` into pure Gaussian noise by
   mixing in a little noise at every step ``t``. Thanks to the Gaussian algebra
   it has a *closed form* — you can jump straight to any timestep ``t`` without
   simulating the steps in between::

       x_t = sqrt(acp_t) * x0 + sqrt(1 - acp_t) * eps,   eps ~ N(0, I)

   where ``acp_t`` (``alpha_bar_t``) is the cumulative product of ``1 - beta``.

2. The *schedule* is just the sequence of ``beta_t`` (how much noise per step).
   Linear (the original DDPM) and cosine (improved DDPM) are provided; swapping
   them is exercise (b).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    """Original DDPM schedule: betas rise linearly from start to end."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Improved-DDPM cosine schedule (Nichol & Dhariwal 2021).

    alpha_bar follows a shifted cosine so noise is added more gently at the
    start and end. We derive betas from alpha_bar and clip for stability.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)


def make_betas(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "linear":
        return linear_beta_schedule(timesteps)
    if schedule == "cosine":
        return cosine_beta_schedule(timesteps)
    raise ValueError(f"unknown schedule {schedule!r} (use 'linear' or 'cosine')")


@dataclass
class Diffusion:
    """Precomputed forward-process constants + the sampling math.

    Build with :meth:`from_schedule`, then use :meth:`q_sample` to noise data
    and :meth:`p_sample_loop` to denoise from pure noise back to samples.
    Everything is vectorised over a batch and a per-example timestep tensor.
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    timesteps: int

    @classmethod
    def from_schedule(cls, schedule: str = "cosine", timesteps: int = 200,
                      device: str | torch.device = "cpu") -> "Diffusion":
        betas = make_betas(schedule, timesteps).to(device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        return cls(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=acp,
            sqrt_alphas_cumprod=torch.sqrt(acp),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - acp),
            timesteps=timesteps,
        )

    def to(self, device) -> "Diffusion":
        return Diffusion(
            self.betas.to(device), self.alphas.to(device),
            self.alphas_cumprod.to(device), self.sqrt_alphas_cumprod.to(device),
            self.sqrt_one_minus_alphas_cumprod.to(device), self.timesteps,
        )

    def _gather(self, vec: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Pick vec[t] per example and reshape to broadcast over data dims."""
        out = vec.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward process: return x_t given clean x0 and timestep t (closed form)."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_acp = self._gather(self.sqrt_alphas_cumprod, t, x0.ndim)
        s_om = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim)
        return s_acp * x0 + s_om * noise

    @torch.no_grad()
    def p_sample_step(self, model, x_t: torch.Tensor, t: int,
                      model_kwargs: dict | None = None) -> torch.Tensor:
        """One reverse (denoising) step of ancestral DDPM sampling."""
        model_kwargs = model_kwargs or {}
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps = model(x_t, t_batch, **model_kwargs)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        acp_t = self.alphas_cumprod[t]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t]

        # mean of p(x_{t-1} | x_t): subtract the predicted noise, rescale.
        mean = (x_t - beta_t / sqrt_om * eps) / torch.sqrt(alpha_t)
        if t == 0:
            return mean
        # add fresh noise scaled by the posterior variance (here: beta_t).
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, model_kwargs: dict | None = None,
                      record_every: int | None = None):
        """Full reverse process from pure noise to a sample.

        Returns ``(x0, trajectory)`` where trajectory is a list of intermediate
        tensors (including start and end) if ``record_every`` is set, else None.
        """
        x = torch.randn(shape, device=device)
        traj = [x.clone()] if record_every else None
        for t in reversed(range(self.timesteps)):
            x = self.p_sample_step(model, x, t, model_kwargs)
            if record_every and (t % record_every == 0 or t == 0):
                traj.append(x.clone())
        return x, traj
