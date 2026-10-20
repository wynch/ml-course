"""Section 1: the bridge. Flow matching against DDPM on module 09's 2-D spiral.

Module 09 trained a network to predict the noise in a noised point (DDPM).
Flow matching (Lipman et al., 2022; guide and code, 2024) trains it to
predict a velocity instead: draw a noise point x0 and a data point x1, put
x_t = (1 - t) x0 + t x1 on the straight line between them, and regress the
network on x1 - x0. Sampling follows the velocity from noise (t = 0) to data
(t = 1) with Euler steps. Both models here share module 09's MLP, data and
optimiser; only the target and the sampler differ.

The spiral, the MLP and the time embedding are copied from module 09
(``src/toy2d.py``, ``src/embeddings.py``); module 09's DDPM is rebuilt here
with continuous time, so both models can be sampled with any number of
steps: DDPM with DDIM's deterministic update (Song et al., 2020) and with
DDPM's own stochastic (ancestral) sampler, which module 09 uses.

    uv run python src/flow2d.py            # trains both, samples, writes results/flow2d.json
"""

from __future__ import annotations

import math

import common
import numpy as np
import torch
import torch.nn as nn

STEPS_LIST = (1, 2, 4, 8, 16, 64, 200)
TRAIN_STEPS, BATCH, LR = 4000, 256, 2e-3
N_EVAL = 4000
X0_CLIP = 3.0         # the spiral spans about +-2.2 after normalising


# ---------------------------------------------------------------- copied from module 09
def make_spiral(n: int = 2000, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0.15, 1.0, n)) * 3.0 * np.pi
    r = t / (3.0 * np.pi)
    pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1) + rng.normal(0, noise, (n, 2))
    pts = pts - pts.mean(axis=0, keepdims=True)
    return (pts / (pts.std(axis=0, keepdims=True) + 1e-8)).astype(np.float32)


class TimeEmbedding(nn.Module):
    """Module 09's sinusoidal embedding; t in [0, 1] is scaled by 1000 to cover the same range."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / (half - 1))
        args = (1000 * t)[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MLP(nn.Module):
    """Module 09's DenoiseMLP without the class label: (x, t) -> a 2-D vector."""

    def __init__(self, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.net = nn.Sequential(nn.Linear(2 + time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_mlp(self.time_embed(t))], -1))


# ---------------------------------------------------------------- the two objectives
def fm_target(x0, x1, t):
    """Flow matching: the point on the line, and the velocity to regress on."""
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    return xt, x1 - x0


def alpha_bar(s):
    """Cosine schedule in continuous time (Nichol and Dhariwal, 2021); s = 0 data, s = 1 noise."""
    f = lambda u: torch.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
    return (f(s) / f(torch.zeros_like(s))).clamp(1e-5, 1.0)


def ddpm_target(x0_noise, x1_data, s):
    """DDPM: noise the data to level s; the target is the noise."""
    ab = alpha_bar(s)[:, None]
    return ab.sqrt() * x1_data + (1 - ab).sqrt() * x0_noise, x0_noise


def train(kind: str, data: torch.Tensor, seed: int = 0, steps: int = TRAIN_STEPS) -> tuple[MLP, list[float]]:
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for _ in range(steps):
        x1 = data[torch.randint(len(data), (BATCH,))]
        x0 = torch.randn_like(x1)
        t = torch.rand(BATCH)
        xt, target = fm_target(x0, x1, t) if kind == "fm" else ddpm_target(x0, x1, t)
        loss = ((model(xt, t) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model.eval(), losses


@torch.no_grad()
def sample_fm(model, z: torch.Tensor, steps: int, keep: bool = False):
    """Euler from t = 0 (noise) to t = 1 (data)."""
    x, path = z.clone(), [z.clone()]
    for i in range(steps):
        t = torch.full((len(x),), i / steps)
        x = x + model(x, t) / steps
        path.append(x.clone())
    return (x, path) if keep else x


@torch.no_grad()
def sample_ddim(model, z: torch.Tensor, steps: int, keep: bool = False):
    """DDIM, deterministic: from noise level 1 down to 0 in equal steps. The data estimate is
    clipped to [-X0_CLIP, X0_CLIP], as DDPM samplers clip images to [-1, 1]: at level 1 the
    cosine schedule keeps no signal, and dividing by sqrt(alpha_bar) there blows the estimate up.
    The noise estimate is then recomputed to match the clipped one, or the two disagree and the
    error grows from step to step."""
    x, path = z.clone(), [z.clone()]
    levels = torch.linspace(1, 0, steps + 1)
    for i in range(steps):
        s, s2 = levels[i].repeat(len(x)), levels[i + 1].repeat(len(x))
        ab, ab2 = alpha_bar(s)[:, None], alpha_bar(s2)[:, None]
        eps = model(x, s)
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-X0_CLIP, X0_CLIP)
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()          # the noise that matches the clipped estimate
        x = ab2.sqrt() * x0 + (1 - ab2).sqrt() * eps
        path.append(x.clone())
    return (x, path) if keep else x


@torch.no_grad()
def sample_ancestral(model, z: torch.Tensor, steps: int, keep: bool = False, seed: int = 3):
    """DDPM's own stochastic sampler, as module 09 runs it, on `steps` equal steps of noise level:
    the posterior mean given the (clipped) data estimate, plus fresh noise of the posterior's
    variance, except on the last step."""
    x, path = z.clone(), [z.clone()]
    g = torch.Generator().manual_seed(seed)
    levels = torch.linspace(1, 0, steps + 1)
    for i in range(steps):
        s, s2 = levels[i].repeat(len(x)), levels[i + 1].repeat(len(x))
        ab, ab2 = alpha_bar(s)[:, None], alpha_bar(s2)[:, None]
        x0 = ((x - (1 - ab).sqrt() * model(x, s)) / ab.sqrt()).clamp(-X0_CLIP, X0_CLIP)
        a = ab / ab2
        mean = (ab2.sqrt() * (1 - a) / (1 - ab)) * x0 + (a.sqrt() * (1 - ab2) / (1 - ab)) * x
        x = mean + ((1 - a) * (1 - ab2) / (1 - ab)).sqrt() * torch.randn(x.shape, generator=g) if i < steps - 1 else mean
        path.append(x.clone())
    return (x, path) if keep else x


def sliced_w1(a: np.ndarray, b: np.ndarray, n_dir: int = 64) -> float:
    """Mean over n_dir fixed directions of the 1-D Wasserstein-1 distance between the projections
    (equal sample sizes: mean absolute gap between sorted projections). 0 = same cloud."""
    ang = np.linspace(0, np.pi, n_dir, endpoint=False)
    D = np.stack([np.cos(ang), np.sin(ang)], 1)
    pa, pb = np.sort(a @ D.T, 0), np.sort(b @ D.T, 0)
    return float(np.abs(pa - pb).mean())


def straightness(path: list[torch.Tensor]) -> float:
    """Total length of the sampled paths over the total distance from start to end; 1 = straight.
    A ratio of sums, so the few points that end near where they started do not dominate."""
    seg = sum((path[i + 1] - path[i]).norm(dim=1) for i in range(len(path) - 1))
    return float(seg.sum() / (path[-1] - path[0]).norm(dim=1).sum())


def main() -> None:
    guard = common.Guard("flow2d", "cpu")
    data = torch.from_numpy(make_spiral(2000, seed=0))
    held = make_spiral(N_EVAL, seed=1)
    models, curves = {}, {}
    for kind in ("fm", "ddpm"):
        models[kind], curves[kind] = train(kind, data)
    guard.tick("train")
    z = torch.randn(N_EVAL, 2, generator=torch.Generator().manual_seed(7))
    rows, shows = {}, {}
    for kind, net, fn in (("fm", "fm", sample_fm), ("ddim", "ddpm", sample_ddim), ("ddpm", "ddpm", sample_ancestral)):
        rows[kind] = {}
        for n in STEPS_LIST:
            x, path = fn(models[net], z, n, keep=True)
            x = x.numpy()
            rows[kind][str(n)] = {"sliced_w1": round(sliced_w1(x, held), 4),
                                  "spread": round(float(x.std(0).mean()), 4),
                                  "straightness": round(straightness(path), 4)}
            shows[f"{kind}_{n}"] = x[:600].round(3).tolist()
        _, path = fn(models[net], z[:24], 64, keep=True)
        shows[f"{kind}_paths"] = torch.stack(path, 1).numpy().round(3).tolist()
    guard.tick("sample")
    out = {
        "what": "flow matching (velocity, Euler) against DDPM (noise; sampled with DDIM and with DDPM's ancestral sampler) on module 09's spiral; same MLP, data, optimiser, seed",
        "train": {"points": 2000, "steps": TRAIN_STEPS, "batch": BATCH, "lr": LR,
                  "final_loss_mean_last200": {k: round(float(np.mean(v[-200:])), 4) for k, v in curves.items()}},
        "eval": {"held_out_points": N_EVAL, "metric": "sliced Wasserstein-1 over 64 directions, against held-out spiral points",
                 "floor_two_spiral_draws": round(sliced_w1(make_spiral(N_EVAL, seed=2), held), 4),
                 "noise_itself": round(sliced_w1(z.numpy(), held), 4)},
        "steps": list(STEPS_LIST),
        "results": rows,
        "loss_curves": {k: [round(float(np.mean(v[i: i + 50])), 4) for i in range(0, len(v), 50)] for k, v in curves.items()},
        "guard": guard.report(),
    }
    common.write_results("flow2d", out)
    common.write_results("flow2d_points", {"held_out": held[:600].round(3).tolist(), **shows})


if __name__ == "__main__":
    main()
