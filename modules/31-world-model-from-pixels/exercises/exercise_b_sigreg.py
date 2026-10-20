"""Exercise (b): SIGReg and LeWM's two-term loss.

SIGReg asks a batch of latents to look like samples of N(0, I). It cannot
test D dimensions at once, so it projects the batch on M random unit
directions and tests each 1-D projection: the projection's empirical
characteristic function, mean over the batch of exp(i t p), should equal the
standard normal's, exp(-t^2 / 2). The squared gap, weighted by exp(-t^2 / 2),
is integrated over t; the gap's real part is symmetric in t and its imaginary
part antisymmetric, so the integral over [0, 3], doubled, covers both signs.

Predict before you run: rank the six latents in main() by their SIGReg, from
smallest to largest. Where do the two latents of a 2-D world land? Fill in
the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_sigreg.py
    uv run pytest tests/test_solutions.py -k sigreg -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import torch
import torch.nn.functional as F


def sigreg(z: torch.Tensor, A: torch.Tensor, knots: int = 17, t_max: float = 3.0) -> torch.Tensor:
    """z: (B, D) latents; A: (D, M) unit columns. Mean over the M projections of the Epps-Pulley
    statistic, times B. Each projection's characteristic function is compared with exp(-t^2 / 2),
    weighted by exp(-t^2 / 2), over t in [0, t_max] by the trapezoid rule, doubled for the negative t."""
    t = torch.linspace(0, t_max, knots)
    dt = t_max / (knots - 1)
    w = torch.full((knots,), 2 * dt)
    w[[0, -1]] = dt
    phi = torch.exp(-t ** 2 / 2)
    # TODO(you): project z on A, multiply by t to get (B, M, knots); take the mean over the batch of cos and
    # of sin; err = (mean cos - phi)^2 + (mean sin)^2; integrate with the weights w * phi (a matrix product);
    # times B; mean over the M directions.
    ...


def lewm_loss(z: torch.Tensor, z_pred: torch.Tensor, A: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
    """z: (B, T, D) encoded frames; z_pred: (B, T - 1, D) the predictor's guesses for frames 1..T-1.
    Mean squared error to the encoded next frames, plus lam times SIGReg averaged over the T frames."""
    # TODO(you): the prediction term (F.mse_loss against the encoded frames 1..T-1), plus lam times the mean
    # over the T frames of sigreg of that frame's (B, D) latents.
    ...


def main() -> None:
    torch.manual_seed(0)
    n, d, m = 512, 32, 1024
    A = torch.randn(d, m)
    A = A / A.norm(dim=0)
    g = torch.randn(n, d)
    xy = torch.rand(n, 2) * 2 - 1                         # a world with two degrees of freedom
    cases = {"N(0, I)": g, "N(0, I) times 0.1": 0.1 * g, "N(0, I) times 3": 3 * g,
             "collapsed to a point": torch.zeros(n, d),
             "2-D square, linearly spread over 32 dims": xy @ torch.randn(2, d),
             "2-D square through a random tanh layer": torch.tanh(xy @ (3 * torch.randn(2, d)))}
    for name, z in cases.items():
        print(f"{name:42s} SIGReg {float(sigreg(z, A)):9.3f}")


if __name__ == "__main__":
    main()
