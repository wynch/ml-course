"""Reference solution, exercise (b): two collapse meters and VICReg's variance term."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch
import torch.nn.functional as F


def std_normalised(z: torch.Tensor) -> float:
    """Std over the batch of the L2-normalised latents, averaged over dims, times sqrt(d)."""
    zn = F.normalize(z, dim=1)
    return float(zn.std(0).mean() * math.sqrt(z.shape[1]))


def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy) of the covariance eigenvalues, normalised to sum to 1."""
    zc = z - z.mean(0)
    ev = torch.linalg.eigvalsh(zc.T @ zc / (len(z) - 1)).clamp_min(0)
    q = ev / ev.sum()
    q = q[q > 0]
    return float(torch.exp(-(q * q.log()).sum()))


def variance_hinge(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg's variance term: mean over dims of max(0, gamma - std), std over the batch."""
    return F.relu(gamma - torch.sqrt(z.var(0) + 1e-4)).mean()


def main() -> None:
    torch.manual_seed(0)
    n, d = 2048, 128
    cases = {
        "isotropic Gaussian": torch.randn(n, d, dtype=torch.float64),
        "a point plus tiny noise": 3 + 1e-3 * torch.randn(n, d, dtype=torch.float64),
        "on a line": 3 * torch.randn(n, 1, dtype=torch.float64) * torch.ones(1, d, dtype=torch.float64),
    }
    for name, z in cases.items():
        print(f"{name:24s} std_norm {std_normalised(z):.3f}  erank {effective_rank(z):6.2f}  "
              f"variance hinge {float(variance_hinge(z)):.3f}")


if __name__ == "__main__":
    main()
