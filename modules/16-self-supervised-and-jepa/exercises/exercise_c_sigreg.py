"""Exercise (c): SIGReg by hand.

SIGReg (Balestriero and LeCun, LeJEPA, 2025) asks latents to look like an
isotropic Gaussian, N(0, I). It cannot test d dimensions at once, so it
projects the latents on K random unit directions and tests each projection
against N(0, 1). The test is Epps-Pulley's: compare the empirical
characteristic function, mean_n exp(i t x_n), with the Gaussian's,
exp(-t^2 / 2), weighted by exp(-t^2 / 2), integrated over t.

Predict before you run: rank these by SIGReg score, lowest first: N(0, I);
N(0, I) scaled by 0.1; every latent equal to zero. Fill in the
``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_c_sigreg.py
    uv run pytest tests/test_solutions.py -k sigreg -v
"""

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
    # TODO(you): the real and imaginary parts of the empirical characteristic function
    # (mean over N of cos and of sin), then torch.trapezoid of the weighted squared gap, dim=-1.
    ...


def sigreg(z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """z: (N, d) latents; A: (d, K) unit columns. Mean Epps-Pulley statistic of the K projections, times N."""
    t = torch.linspace(-5, 5, 17)
    # TODO(you): one line.
    ...


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
