"""The gridjepa encoder, copied so this module runs on its own.

gridjepa is the author's earlier experiment: a JEPA world model trained on a
small tile world (an arena of 12 x 12 cells ringed by walls, ten tile kinds,
one agent). The model sees a 15 x 15 window of tiles centred on the agent and
maps it to a 128-number latent z. It was trained only to predict the next z
from the current z and the action (EMA target encoder, stop-gradient); no
label of any kind entered training.

This file holds the encoder class (same layers, same order, so the same seed
gives the same random weights), the one-hot input builder, and the probe
helpers (closed-form ridge, short logistic regression, episode-level split).
The weights are in ../../artifacts/gridjepa_h1_encoder.bin.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ARTIFACTS

N_TILES = 10            # empty, wall, block, key, door, hazard, goal, 3 unused
TILE_NAMES = ["empty", "wall", "block", "key", "door", "hazard", "goal", "red", "blue", "ice"]
K = 15                  # window side
LATENT = 128


class Encoder(nn.Module):
    """15 x 15 tile window (+ 5 scalars) -> z in R^128. About 0.36M parameters."""

    def __init__(self, K: int = K, latent_dim: int = LATENT, n_scalars: int = 5):
        super().__init__()
        in_ch = N_TILES + 1  # ten one-hot tile channels and an agent mask
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
        )
        with torch.no_grad():
            conv_flat = self.conv(torch.zeros(1, in_ch, K, K)).flatten(1).shape[1]
        self.scalar_fc = nn.Linear(n_scalars, 16)
        self.head = nn.Linear(conv_flat + 16, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        feat = self.conv(grid).flatten(1)
        s = F.gelu(self.scalar_fc(scalars))
        return self.norm(self.head(torch.cat([feat, s], dim=1)))


def windows_to_grid(win: torch.Tensor, K: int = K) -> torch.Tensor:
    """int windows (B, K, K) -> float (B, 11, K, K): one-hot tiles plus the agent mask."""
    onehot = F.one_hot(win.long(), num_classes=N_TILES).permute(0, 3, 1, 2).float()
    mask = torch.zeros(win.shape[0], 1, K, K)
    mask[:, 0, K // 2, K // 2] = 1.0
    return torch.cat([onehot, mask], dim=1)


def load_trained() -> Encoder:
    enc = Encoder()
    enc.load_state_dict(torch.load(ARTIFACTS / "gridjepa_h1_encoder.bin", map_location="cpu",
                                   weights_only=True))
    return enc.eval()


def load_random(seed: int) -> Encoder:
    torch.manual_seed(seed)
    return Encoder().eval()


def load_states() -> dict:
    """10,000 arena frames: windows, scalars, agent position, BFS steps to goal, episode id."""
    d = np.load(ARTIFACTS / "grid_states.npz")
    return {k: d[k] for k in d.files}


@torch.no_grad()
def encode(enc: Encoder, windows: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    enc.eval()
    out = []
    for i in range(0, len(windows), 1024):
        w = torch.from_numpy(windows[i:i + 1024]).long()
        s = torch.from_numpy(scalars[i:i + 1024]).float()
        out.append(enc(windows_to_grid(w), s).numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ probes
def ridge_fit(Z: np.ndarray, Y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Closed-form ridge with a bias column. Returns W of shape (D + 1, out)."""
    n, d = Z.shape
    Zb = np.concatenate([Z, np.ones((n, 1), dtype=Z.dtype)], axis=1)
    return np.linalg.solve(Zb.T @ Zb + lam * np.eye(d + 1), Zb.T @ Y)


def ridge_predict(W: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.concatenate([Z, np.ones((len(Z), 1), dtype=Z.dtype)], axis=1) @ W


def r2_score(Yhat: np.ndarray, Y: np.ndarray) -> float:
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def logreg_fit_predict(Ztr, ytr, Zte, n_classes: int, steps: int = 300, seed: int = 0) -> np.ndarray:
    """300 Adam steps of multinomial logistic regression; predicted classes for Zte."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(Ztr.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr, Ytr = torch.from_numpy(Ztr).float(), torch.from_numpy(ytr).long()
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(Xtr), Ytr).backward()
        opt.step()
    with torch.no_grad():
        return lin(torch.from_numpy(Zte).float()).argmax(dim=1).numpy()


def split_by_episode(ep: np.ndarray, frac_train: float = 0.7, seed: int = 0):
    """Whole episodes go to one side: consecutive frames are near-copies."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(ep))
    train_eps = set(uniq[:max(1, int(len(uniq) * frac_train))].tolist())
    is_tr = np.fromiter((e in train_eps for e in ep), dtype=bool, count=len(ep))
    return np.where(is_tr)[0], np.where(~is_tr)[0]


def adjacency_labels(windows: np.ndarray) -> np.ndarray:
    """Tile id at the 8 cells around the agent, (N, 8)."""
    c = K // 2
    offs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    labs = np.empty((len(windows), 8), dtype=np.int64)
    for j, (dx, dy) in enumerate(offs):
        labs[:, j] = windows[:, c + dy, c + dx]
    return labs
