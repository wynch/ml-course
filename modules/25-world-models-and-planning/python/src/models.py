"""Two models of the grid world, their training, and the probe that reads a latent.

- ``OneHotModel``: observation + action -> a score for each of the 64 cells
  where the agent may be next. Walls and goal do not move, so the next
  observation is the old one with the agent plane replaced by the cell with
  the highest score.
- ``JEPA``: an encoder maps the observation to a 32-number latent; a predictor
  maps (latent, action) to the next latent. Module 16's objective: the target
  is what a slow moving average of the encoder (EMA) outputs for the next
  observation, with no gradient; the loss is the squared error between latents.
  The model never sees a pixel, cell or wall it must rebuild.

A JEPA latent means nothing to a planner until something reads it. ``Probe``
is a ridge regression from the latent to the goal's offset from the agent,
fitted on training observations (fit it in the space the
predictor writes to, the target encoder's).
"""

from __future__ import annotations

import copy
import math

import common  # noqa: F401  (laptop guard before torch)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grid import N, OBS_DIM

LATENT = 32
HIDDEN = 128
STEPS = 6000
BATCH = 256
LR = 1e-3
EMA_START, EMA_END = 0.996, 1.0      # module 16's values; module 16 ramps linearly, this ramps by cosine
RIDGE = 1e-2


# ------------------------------------------------------------------ models
class OneHotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM + 4, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, N * N))

    def forward(self, obs, action):
        return self.net(torch.cat([obs, F.one_hot(action, 4).float()], -1))     # (B, 64) logits

    @torch.no_grad()
    def rollout(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs (B, 192), actions (B, H) -> the agent cell after each step, (B, H, 2)."""
        obs = obs.clone()
        cells = []
        for k in range(actions.shape[1]):
            idx = self(obs, actions[:, k]).argmax(-1)
            agent = torch.zeros(len(obs), N * N)
            agent[torch.arange(len(obs)), idx] = 1
            obs[:, N * N:2 * N * N] = agent
            cells.append(torch.stack([idx // N, idx % N], -1))
        return torch.stack(cells, 1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT))

    def forward(self, obs):
        return self.net(obs)


class Predictor(nn.Module):
    """z' = z + f(z, a): the action moves the latent; no action, no move is the easy default."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT + 4, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT))

    def forward(self, z, action):
        return z + self.net(torch.cat([z, F.one_hot(action, 4).float()], -1))


class JEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def rollout(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs (B, 192), actions (B, H) -> the imagined latent after each step, (B, H, LATENT).

        Starts from the target encoder, the space the predictor was trained to write.
        """
        z = self.target(obs)
        out = []
        for k in range(actions.shape[1]):
            z = self.predictor(z, actions[:, k])
            out.append(z)
        return torch.stack(out, 1)


def jepa_loss(model: JEPA, obs, action, next_obs) -> torch.Tensor:
    """Squared error between the predicted latent and the EMA target's latent of the next observation."""
    pred = model.predictor(model.encoder(obs), action)
    with torch.no_grad():
        tgt = model.target(next_obs)
    return ((pred - tgt) ** 2).mean()


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, mom: float) -> None:
    for a, b in zip(target.parameters(), online.parameters()):
        a.mul_(mom).add_(b.detach(), alpha=1 - mom)


def ema_momentum(step: int, total: int) -> float:
    return EMA_END - (EMA_END - EMA_START) * (math.cos(math.pi * step / total) + 1) / 2


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------- training
def tensors(data: dict) -> dict:
    return {"obs": torch.from_numpy(data["obs"]), "action": torch.from_numpy(data["action"]).long(),
            "next_obs": torch.from_numpy(data["next_obs"]),
            "next_idx": torch.from_numpy(data["next_pos"][:, 0] * N + data["next_pos"][:, 1]).long()}


def train_onehot(data: dict, seed: int = 0, steps: int = STEPS, log_every: int = 200):
    torch.manual_seed(seed)
    model = OneHotModel()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    T = tensors(data)
    g = torch.Generator().manual_seed(seed)
    log = []
    for s in range(1, steps + 1):
        i = torch.randint(len(T["obs"]), (BATCH,), generator=g)
        loss = F.cross_entropy(model(T["obs"][i], T["action"][i]), T["next_idx"][i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % log_every == 0:
            log.append({"step": s, "loss": round(float(loss), 5)})
    return model.eval(), log


def train_jepa(data: dict, seed: int = 0, steps: int = STEPS, log_every: int = 200):
    torch.manual_seed(seed)
    model = JEPA()
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=LR)
    T = tensors(data)
    g = torch.Generator().manual_seed(seed)
    log = []
    for s in range(1, steps + 1):
        i = torch.randint(len(T["obs"]), (BATCH,), generator=g)
        loss = jepa_loss(model, T["obs"][i], T["action"][i], T["next_obs"][i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update(model.target, model.encoder, ema_momentum(s, steps))
        if s % log_every == 0:
            log.append({"step": s, "loss": round(float(loss), 6)})
    return model.eval(), log


def random_jepa(seed: int = 0) -> JEPA:
    """The same JEPA at its initial random weights: the control."""
    torch.manual_seed(seed)
    return JEPA().eval()


# ------------------------------------------------------------------- probe
class Probe:
    """Ridge regression with bias, closed form: latent -> goal offset (goal row - agent row, goal col - agent col)."""

    def __init__(self, W: np.ndarray):
        self.W = W                                    # (LATENT + 1, 2)

    @classmethod
    def fit(cls, Z: np.ndarray, Y: np.ndarray, lam: float = RIDGE) -> "Probe":
        Zb = np.concatenate([Z, np.ones((len(Z), 1))], 1).astype(np.float64)
        W = np.linalg.solve(Zb.T @ Zb + lam * np.eye(Zb.shape[1]), Zb.T @ Y.astype(np.float64))
        return cls(W)

    def __call__(self, Z):
        if isinstance(Z, torch.Tensor):
            W = torch.from_numpy(self.W).float()
            return Z @ W[:-1] + W[-1]
        return Z @ self.W[:-1] + self.W[-1]


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


@torch.no_grad()
def target_latents(model: JEPA, obs: np.ndarray) -> np.ndarray:
    return model.target(torch.from_numpy(obs)).numpy()


def fit_probe(model: JEPA, data: dict, n: int = 20_000) -> Probe:
    Z = target_latents(model, data["obs"][:n])
    return Probe.fit(Z, (data["goal"] - data["pos"])[:n])


# ---------------------------------------------------------- collapse meters
@torch.no_grad()
def collapse_meters(z: np.ndarray) -> dict:
    """Module 16's two meters: normalised std (about 1 = spread over the sphere, 0 = one direction)
    and effective rank (exp of the entropy of the covariance eigenvalues; LATENT at most)."""
    z = torch.from_numpy(np.asarray(z, dtype=np.float64))
    zn = F.normalize(z, dim=1)
    std_norm = float(zn.std(0).mean() * math.sqrt(z.shape[1]))
    zc = z - z.mean(0)
    ev = torch.linalg.eigvalsh(zc.T @ zc / (len(z) - 1)).clamp_min(0)
    q = ev / ev.sum()
    q = q[q > 0]
    return {"std_norm": round(std_norm, 4), "erank": round(float(torch.exp(-(q * q.log()).sum())), 2)}


# ------------------------------------------------------------- persistence
def save(models: dict, probes: dict, path) -> None:
    torch.save({"state": {k: m.state_dict() for k, m in models.items()},
                "probes": {k: p.W for k, p in probes.items()}}, path)


def load(path) -> tuple[dict, dict]:
    ck = torch.load(path, weights_only=False)
    models = {}
    for k, sd in ck["state"].items():
        m = OneHotModel() if k == "onehot" else JEPA()
        m.load_state_dict(sd)
        models[k] = m.eval()
    return models, {k: Probe(W) for k, W in ck["probes"].items()}
