"""The parts every section shares: the encoder, augmentations, masks, losses,
anti-collapse terms, collapse meters and linear probes.

Images enter as uint8 (N, 32, 32, 3) on the device; ``to_float`` turns them
into normalised float (N, 3, 32, 32).
"""

from __future__ import annotations

import math

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 128           # latent width per grid cell, and of the pooled vector the probes read
GRID = 8          # the encoder's output map is GRID x GRID; one cell sees a 4x4 pixel patch at its centre
MEAN = (0.491, 0.482, 0.447)
STD = (0.247, 0.243, 0.262)


# ------------------------------------------------------------------ encoder
def _block(cin, cout, stride=1):
    return [nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.ReLU()]


class Encoder(nn.Module):
    """32x32x3 -> an 8x8 map of D-dim latents; ``forward`` average-pools it.

    GroupNorm, not BatchNorm: BatchNorm mixes statistics across the batch,
    and that alone can keep a self-supervised model from collapsing, which
    would hide what section 2 shows. The last layer has no norm, so the
    latents are free to shrink to a point.
    """

    def __init__(self, d: int = D):
        super().__init__()
        self.net = nn.Sequential(*_block(3, 32), *_block(32, 64, 2), *_block(64, 64),
                                 *_block(64, 128, 2), *_block(128, 128), nn.Conv2d(128, d, 1))

    def forward_map(self, x):
        return self.net(x)                                       # (B, D, 8, 8)

    def forward(self, x):
        return self.net(x).mean((2, 3))                          # (B, D)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------ augmentations
def to_float(xb: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 32, 32, 3) -> float (B, 3, 32, 32) in [0, 1]."""
    return xb.permute(0, 3, 1, 2).float() / 255


def normalize(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def random_resized_crop(x: torch.Tensor, scale=(0.2, 1.0), flip=True, gen=None) -> torch.Tensor:
    """A crop of `scale` of the area, aspect 3/4 to 4/3, resized back to 32x32, maybe flipped; one per image."""
    B, dev = x.shape[0], x.device
    u = lambda: torch.rand(B, device=dev, generator=gen)
    area = scale[0] + (scale[1] - scale[0]) * u()
    logr = math.log(3 / 4) + (math.log(4 / 3) - math.log(3 / 4)) * u()
    ar = torch.exp(logr)
    w = torch.sqrt(area * ar).clamp(max=1.0)
    h = torch.sqrt(area / ar).clamp(max=1.0)
    cx = (1 - w) * (2 * u() - 1)
    cy = (1 - h) * (2 * u() - 1)
    sx = w * (torch.where(u() < 0.5, -1.0, 1.0) if flip else 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = sx, cx, h, cy
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def color_distort(x: torch.Tensor, s: float = 0.4, p_jitter=0.8, p_gray=0.2, gen=None) -> torch.Tensor:
    """SimCLR's colour jitter without hue: brightness, contrast, saturation, then grayscale. x in [0, 1]."""
    B, dev = x.shape[0], x.device
    u = lambda: torch.rand(B, 1, 1, 1, device=dev, generator=gen)
    on = (u() < p_jitter).float()
    f = lambda: 1 + on * s * (2 * u() - 1)
    x = x * f()                                                              # brightness
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * f() + m                                                    # contrast
    g = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - g) * f() + g                                                    # saturation
    gray = (u() < p_gray).float()
    g = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = gray * g.expand_as(x) + (1 - gray) * x
    return x.clamp(0, 1)


def simclr_view(xb: torch.Tensor, gen=None) -> torch.Tensor:
    return normalize(color_distort(random_resized_crop(to_float(xb), (0.2, 1.0), gen=gen), gen=gen))


def light_view(xb: torch.Tensor, gen=None) -> torch.Tensor:
    """Crop (60-100% of the area) and flip: the only augmentation of the other objectives."""
    return normalize(random_resized_crop(to_float(xb), (0.6, 1.0), gen=gen))


# -------------------------------------------------------------------- masks
def random_patch_mask(B: int, ratio: float, dev, gen=None) -> torch.Tensor:
    """(B, 8, 8) bool, True = hidden; exactly round(ratio * 64) cells per image."""
    k = round(ratio * GRID * GRID)
    idx = torch.rand(B, GRID * GRID, device=dev, generator=gen).argsort(1)[:, :k]
    m = torch.zeros(B, GRID * GRID, dtype=torch.bool, device=dev)
    return m.scatter_(1, idx, True).view(B, GRID, GRID)


def block_mask(B: int, rng: np.random.Generator, dev, n_blocks=4, scale=(0.15, 0.2), aspect=(0.75, 1.5)) -> torch.Tensor:
    """I-JEPA's target mask on the 8x8 grid: the union of n_blocks rectangles of 15-20% each. (B, 8, 8) bool."""
    area = rng.uniform(*scale, size=(B, n_blocks)) * GRID * GRID
    ar = np.exp(rng.uniform(math.log(aspect[0]), math.log(aspect[1]), size=(B, n_blocks)))
    h = np.clip(np.rint(np.sqrt(area * ar)), 1, GRID).astype(int)
    w = np.clip(np.rint(np.sqrt(area / ar)), 1, GRID).astype(int)
    y0 = (rng.random((B, n_blocks)) * (GRID - h + 1)).astype(int)
    x0 = (rng.random((B, n_blocks)) * (GRID - w + 1)).astype(int)
    yy = np.arange(GRID)[None, None, :, None]
    xx = np.arange(GRID)[None, None, None, :]
    inside = ((yy >= y0[..., None, None]) & (yy < (y0 + h)[..., None, None])
              & (xx >= x0[..., None, None]) & (xx < (x0 + w)[..., None, None]))
    return torch.from_numpy(inside.any(1)).to(dev)


def apply_mask(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Zero (the dataset mean, after normalising) every 4x4 pixel patch whose grid cell is masked."""
    up = m.repeat_interleave(4, 1).repeat_interleave(4, 2).unsqueeze(1)    # (B, 1, 32, 32)
    return x.masked_fill(up, 0.0)


# ------------------------------------------------------------------- losses
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """SimCLR's loss: each view must pick its partner out of the 2B - 1 other views in the batch."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    B = z1.shape[0]
    sim = z @ z.T / tau
    sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
    target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, target)


class Predictor(nn.Module):
    """JEPA's predictor: from the context map plus a 'predict here' channel, a latent for every cell."""

    def __init__(self, d: int = D, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(d + 1, h, 3, 1, 1), nn.GroupNorm(8, h), nn.ReLU(),
                                 nn.Conv2d(h, h, 3, 1, 1), nn.GroupNorm(8, h), nn.ReLU(), nn.Conv2d(h, d, 1))

    def forward(self, ctx_map, tgt_mask):
        return self.net(torch.cat([ctx_map, tgt_mask.unsqueeze(1).float()], 1))


class PixelDecoder(nn.Module):
    """Masked reconstruction's decoder: 8x8 latents back to 32x32x3 pixels."""

    def __init__(self, d: int = D, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(d, h, 4, 2, 1), nn.GroupNorm(8, h), nn.ReLU(),
                                 nn.ConvTranspose2d(h, h, 4, 2, 1), nn.GroupNorm(8, h), nn.ReLU(), nn.Conv2d(h, 3, 3, 1, 1))

    def forward(self, m):
        return self.net(m)


def jepa_loss(pred_map, tgt_map, tgt_mask):
    """Mean squared error over the target cells only."""
    m = tgt_mask.unsqueeze(1).float()
    return (((pred_map - tgt_map) ** 2) * m).sum() / (m.sum() * pred_map.shape[1])


# ------------------------------------------------------ anti-collapse terms
def vicreg_variance(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg's variance term alone: a hinge that pushes each latent's std over the batch up to gamma."""
    std = torch.sqrt(z.var(0) + 1e-4)
    return F.relu(gamma - std).mean()


def sigreg(z: torch.Tensor, K: int = 256, gen=None) -> torch.Tensor:
    """SIGReg (Balestriero and LeCun, LeJEPA, 2025): project z on K random unit directions and test each
    1-D projection against N(0, 1) with the Epps-Pulley statistic, which compares the empirical
    characteristic function with exp(-t^2 / 2) over t in [-5, 5]. The mean statistic, times the batch
    size, is the loss; it is zero only when every projection looks Gaussian with unit variance."""
    A = torch.randn(z.shape[1], K, device=z.device, generator=gen)
    A = A / A.norm(dim=0, keepdim=True)
    pz = (z @ A).unsqueeze(-1)                                               # (B, K, 1)
    t = torch.linspace(-5, 5, 17, device=z.device)
    w = torch.exp(-t ** 2 / 2)
    re, im = torch.cos(pz * t).mean(0), torch.sin(pz * t).mean(0)            # (K, 17)
    return torch.trapezoid(((re - w) ** 2 + im ** 2) * w, t, dim=-1).mean() * len(z)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, mom: float) -> None:
    for a, b in zip(target.parameters(), online.parameters()):
        a.mul_(mom).add_(b.detach(), alpha=1 - mom)


# ------------------------------------------------------------ collapse meters
@torch.no_grad()
def collapse_meters(z: torch.Tensor) -> dict:
    """On pooled latents (N, D): the std of the L2-normalised latents times sqrt(D) (about 1 when the
    latents spread over the sphere, 0 when they all point one way); the raw per-dim std; and the
    effective rank, exp(entropy) of the covariance eigenvalues (D when all directions carry equal
    variance, 1 when one does)."""
    z = z.double().cpu()
    zn = F.normalize(z, dim=1)
    std_norm = float(zn.std(0).mean() * math.sqrt(z.shape[1]))
    zc = z - z.mean(0)
    ev = torch.linalg.eigvalsh(zc.T @ zc / (len(z) - 1)).clamp_min(0)
    if ev.sum() <= 1e-20:
        erank = 1.0
    else:
        q = ev / ev.sum()
        q = q[q > 0]
        erank = float(torch.exp(-(q * q.log()).sum()))
    return {"std_norm": round(std_norm, 4), "std_raw": round(float(z.std(0).mean()), 6), "erank": round(erank, 2)}


# ---------------------------------------------------------------- probes
@torch.no_grad()
def embed(enc: nn.Module, X: torch.Tensor, bs: int = 1000, rot: int = 0) -> torch.Tensor:
    """Pooled latents of uint8 images X (on the device), returned on the CPU; rot = quarter turns."""
    enc.eval()
    out = []
    for i in range(0, len(X), bs):
        x = normalize(to_float(X[i:i + bs]))
        if rot:
            x = torch.rot90(x, rot, (2, 3))
        out.append(enc(x).float().cpu())
    enc.train()
    return torch.cat(out)


def logistic_probe(Ztr, ytr, Zte, yte, n_cls: int, epochs: int = 100, seed: int = 0) -> float:
    """A linear classifier on frozen, standardised features; CPU, so it is deterministic. Test accuracy."""
    g = torch.Generator().manual_seed(seed)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    lin = nn.Linear(Ztr.shape[1], n_cls)
    with torch.no_grad():
        lin.weight.normal_(0, 0.01, generator=g)
        lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), 1e-2, weight_decay=1e-5)
    for _ in range(epochs):
        for i in torch.randperm(len(Ztr), generator=g).split(4096):
            opt.zero_grad()
            F.cross_entropy(lin(Ztr[i]), ytr[i]).backward()
            opt.step()
    with torch.no_grad():
        return float((lin(Zte).argmax(1) == yte).float().mean())


def ridge_r2(Ztr, Ttr, Zte, Tte, lam: float = 1.0) -> float:
    """Ridge regression from standardised features to targets T (N, k); mean R² over the k targets, on test."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    A, B = (Ztr - mu) / sd, (Zte - mu) / sd
    A = torch.cat([A, torch.ones(len(A), 1)], 1).double()
    B = torch.cat([B, torch.ones(len(B), 1)], 1).double()
    reg = lam * torch.eye(A.shape[1], dtype=torch.float64)
    reg[-1, -1] = 0
    W = torch.linalg.solve(A.T @ A + reg, A.T @ Ttr.double())
    P = B @ W
    T = Tte.double()
    r2 = 1 - ((T - P) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)
    return float(r2.mean())


def all_probes(enc: nn.Module, Xtr, ytr, Xte, yte, n_rot_train: int = 10000) -> dict:
    """Section 1's class probe (50,000 and 10,000 training images) and section 3's colour and rotation probes."""
    Ztr, Zte = embed(enc, Xtr), embed(enc, Xte)
    ytr_c, yte_c = ytr.cpu(), yte.cpu()
    out = {"class_acc": logistic_probe(Ztr, ytr_c, Zte, yte_c, 10),
           "class_acc_10k": logistic_probe(Ztr[:10000], ytr_c[:10000], Zte, yte_c, 10)}
    col_tr = to_float(Xtr).mean((2, 3)).cpu()                               # mean R, G, B in [0, 1]
    col_te = to_float(Xte).mean((2, 3)).cpu()
    out["colour_r2"] = ridge_r2(Ztr, col_tr, Zte, col_te)
    Xr = Xtr[:n_rot_train]
    Rtr = torch.cat([embed(enc, Xr, rot=k) for k in range(4)])
    Rte = torch.cat([embed(enc, Xte, rot=k) for k in range(4)])
    rtr = torch.arange(4).repeat_interleave(len(Xr))
    rte = torch.arange(4).repeat_interleave(len(Xte))
    out["rotation_acc"] = logistic_probe(Rtr, rtr, Rte, rte, 4)
    out["meters_test"] = collapse_meters(Zte)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
