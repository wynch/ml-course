"""A LeWorldModel in miniature: encoder, predictor, two losses, no EMA, no stop-gradient.

- ``Encoder``: four strided convolutions (64 -> 32 -> 16 -> 8 -> 4 pixels) and
  a linear layer to 256 features, then a projector (linear, BatchNorm, GELU,
  linear) to the embedding z of D numbers. LeWM puts the same kind of
  projector after its ViT, and says it needs the BatchNorm.
- ``Predictor``: (z_t, one block of 5 actions) -> z_{t+1}. It does not add its
  output to z_t: module 25's predictor did, and that made "nothing moves" free.
- Loss (LeWM, eq. 3): mean squared error between predicted and encoded next
  embeddings, plus lambda * SIGReg of all embeddings. Gradients flow through
  both sides; the target is the same encoder, not a slow copy.

``sigreg`` follows LeJEPA and the stable-worldmodel code: project the batch on
M random unit directions, compare each 1-D projection's characteristic
function with the standard normal's, exp(-t^2/2), weighted by exp(-t^2/2),
integrated over t in [0, 3] by the trapezoid rule (doubled, by symmetry),
times the batch size.
"""

from __future__ import annotations

import math

import common  # noqa: F401  (laptop guard before torch)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tworoom as W

BLOCK = 5                      # environment steps per model step (LeWM's frameskip)
ACT_IN = BLOCK * W.ACT_DIM     # 10 numbers per action block
FEAT = 256
HIDDEN = 256
SEQ = 4                        # frames per training sub-trajectory (3 transitions)
KNOTS, T_MAX, M_PROJ = 17, 3.0, 1024


# -------------------------------------------------------------------- SIGReg
def sigreg(z: torch.Tensor, A: torch.Tensor | None = None, M: int = M_PROJ, gen=None) -> torch.Tensor:
    """z: (B, D) or (T, B, D). Mean over directions (and time) of the Epps-Pulley statistic, times B.

    A: (D, M) directions; drawn here, unit length, when not given.
    """
    if A is None:
        A = torch.randn(z.shape[-1], M, device=z.device, generator=gen)
        A = A / A.norm(dim=0)
    t = torch.linspace(0, T_MAX, KNOTS, device=z.device)
    dt = T_MAX / (KNOTS - 1)
    w = torch.full((KNOTS,), 2 * dt, device=z.device)
    w[[0, -1]] = dt
    phi = torch.exp(-t ** 2 / 2)
    xt = (z @ A).unsqueeze(-1) * t                                     # (..., B, M, KNOTS)
    err = (xt.cos().mean(-3) - phi) ** 2 + xt.sin().mean(-3) ** 2      # (..., M, KNOTS)
    return ((err @ (w * phi)) * z.shape[-2]).mean()


# -------------------------------------------------------------------- models
def mlp_bn(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.BatchNorm1d(h), nn.GELU(), nn.Linear(h, o))


class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        c = (3, 32, 64, 64, 128)
        layers = []
        for a, b in zip(c[:-1], c[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.ReLU()]
        self.conv = nn.Sequential(*layers, nn.Flatten(), nn.Linear(128 * 4 * 4, FEAT))
        self.proj = mlp_bn(FEAT, 2 * FEAT, dim)

    def forward(self, x):                      # (B, 3, 64, 64) -> (B, dim)
        return self.proj(self.conv(x))


class Predictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.Sequential(nn.Linear(ACT_IN, 64), nn.SiLU(), nn.Linear(64, 64))
        self.net = nn.Sequential(nn.Linear(dim + 64, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.GELU())
        self.proj = mlp_bn(HIDDEN, 2 * HIDDEN, dim)

    def forward(self, z, a):                   # (B, dim), (B, 10) -> (B, dim)
        return self.proj(self.net(torch.cat([z, self.act(a)], -1)))


class WorldModel(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.encoder = Encoder(dim)
        self.predictor = Predictor(dim)

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z0 (B, dim), actions (B, H, 10) -> imagined embeddings after each block, (B, H, dim)."""
        return rollout(self, z0, actions)


def rollout(model: WorldModel, z0, actions):
    z, out = z0, []
    for k in range(actions.shape[1]):
        z = model.predictor(z, actions[:, k])
        out.append(z)
    return torch.stack(out, 1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------- data
def sample_batch(data: dict, rng: np.random.Generator, batch: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """batch sub-trajectories: frames (batch, SEQ, 3, 64, 64), action blocks (batch, SEQ-1, 10), positions."""
    P, A = data["pos"], data["action"]
    E, L = P.shape[0], A.shape[1]
    e = rng.integers(E, size=batch)
    t0 = rng.integers(L - BLOCK * (SEQ - 1) + 1, size=batch)
    ts = t0[:, None] + BLOCK * np.arange(SEQ)                               # frame times
    pos = P[e[:, None], ts]                                                 # (batch, SEQ, 2)
    at = ts[:, :-1, None] + np.arange(BLOCK)                                # (batch, SEQ-1, BLOCK)
    act = A[e[:, None, None], at].reshape(batch, SEQ - 1, ACT_IN)
    frames = W.render(pos.reshape(-1, 2)).view(batch, SEQ, 3, W.PX, W.PX)
    return frames, torch.from_numpy(act), pos


def losses(model: WorldModel, frames, act, lam: float, gen=None) -> dict:
    """LeWM's loss on a batch of sub-trajectories; returns its two parts and the total."""
    B, T = frames.shape[:2]
    z = model.encoder(frames.flatten(0, 1)).view(B, T, -1)
    pred = model.predictor(z[:, :-1].flatten(0, 1), act.flatten(0, 1)).view(B, T - 1, -1)
    lp = F.mse_loss(pred, z[:, 1:])
    ls = sigreg(z.transpose(0, 1), gen=gen) if lam > 0 else torch.zeros((), device=z.device)
    return {"pred": lp, "sigreg": ls, "total": lp + lam * ls}


@torch.no_grad()
def recalibrate(model: WorldModel, data: dict, batches: int = 50, device="cpu", seed: int = 0) -> WorldModel:
    """Recompute every BatchNorm's running mean and variance on the final weights.

    The features before each BatchNorm have a tiny spread (about 3e-4 variance
    here), so the running averages, gathered while the weights still moved,
    sit far off in units of that spread; in eval mode the embeddings were then
    off by half their size. An even average over `batches` training batches,
    with the weights frozen, fixes it.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None                        # cumulative average
    model.train()
    rng = np.random.default_rng(seed)
    for _ in range(batches):
        f, a, _ = sample_batch(data, rng, 128)
        losses(model, f.to(device), a.to(device), 0.0)
    for bn in bns:
        bn.momentum = 0.1
    return model.eval()


# --------------------------------------------------------------- read-outs
@torch.no_grad()
def embed(model: WorldModel, pos: np.ndarray, device="cpu", chunk: int = 2048) -> np.ndarray:
    model.eval()
    out = [model.encoder(W.render(pos[i:i + chunk]).to(device)).cpu() for i in range(0, len(pos), chunk)]
    return torch.cat(out).numpy().astype(np.float64)


def ridge(Ztr, Ytr, lam: float = 1e-3):
    """Closed-form ridge with bias; returns the (D + 1, k) weights."""
    Zb = np.concatenate([Ztr, np.ones((len(Ztr), 1))], 1)
    return np.linalg.solve(Zb.T @ Zb + lam * np.eye(Zb.shape[1]), Zb.T @ Ytr)


def apply(Wt, Z):
    return Z @ Wt[:-1] + Wt[-1]


def r2(pred, y) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def collapse_meters(z: np.ndarray) -> dict:
    """Module 16's two meters: normalised std (about 1 = spread, 0 = one direction) and effective rank."""
    z = np.asarray(z, np.float64)
    zn = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    std_norm = float(zn.std(0, ddof=1).mean() * math.sqrt(z.shape[1]))
    zc = z - z.mean(0)
    ev = np.clip(np.linalg.eigvalsh(zc.T @ zc / (len(z) - 1)), 0, None)
    if ev.sum() <= 1e-20:
        return {"std_norm": round(std_norm, 4), "erank": 1.0}
    q = ev / ev.sum()
    q = q[q > 0]
    return {"std_norm": round(std_norm, 4), "erank": round(float(np.exp(-(q * np.log(q)).sum())), 2)}


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------- persistence
def save(model: WorldModel, path, extra: dict | None = None, half: bool = False) -> None:
    sd = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"dim": model.dim, "state": sd, **(extra or {})}, path)


def load(path) -> tuple[WorldModel, dict]:
    ck = torch.load(path, weights_only=False, map_location="cpu")
    m = WorldModel(ck["dim"])
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["state"].items()})
    return m.eval(), ck
