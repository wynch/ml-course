"""Reference solution, exercise (c): SIGReg by hand."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch


def epps_pulley(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """p: (N, K) samples of K one-dimensional variables. For each, the integral over t of
    |mean_n exp(i t p_n) - exp(-t^2 / 2)|^2 * exp(-t^2 / 2), by the trapezoid rule. Returns (K,)."""
    w = torch.exp(-t ** 2 / 2)
    pt = p.unsqueeze(-1) * t                          # (N, K, T)
    re, im = torch.cos(pt).mean(0), torch.sin(pt).mean(0)
    return torch.trapezoid(((re - w) ** 2 + im ** 2) * w, t, dim=-1)


def sigreg(z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """z: (N, d) latents; A: (d, K) unit columns. Mean Epps-Pulley statistic of the K projections, times N."""
    t = torch.linspace(-5, 5, 17)
    return epps_pulley(z @ A, t).mean() * len(z)


def main() -> None:
    torch.manual_seed(0)
    n, d, K = 1024, 64, 256
    A = torch.randn(d, K)
    A = A / A.norm(dim=0, keepdim=True)
    g = torch.randn(n, d)
    cases = {"N(0, I)": g, "N(0, I) scaled by 0.1": 0.1 * g, "N(0, I) scaled by 3": 3 * g,
             "N(0, I) shifted by 1": g + 1, "collapsed to a point": torch.zeros(n, d),
             "on a line": torch.randn(n, 1) * torch.ones(1, d) / d ** 0.5}
    for name, z in cases.items():
        print(f"{name:24s} SIGReg {float(sigreg(z, A)):8.3f}")


if __name__ == "__main__":
    main()
