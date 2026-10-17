"""Exercise (b): two collapse meters and VICReg's variance term.

1. ``std_normalised``: L2-normalise each latent, take the std over the batch of
   each dim, average over dims, multiply by sqrt(d). About 1 when the latents
   spread over the sphere, 0 when they all point one way.
2. ``effective_rank``: the covariance's eigenvalues, normalised to sum to 1,
   then exp(entropy). d when every direction carries the same variance, 1
   when one direction carries it all.
3. ``variance_hinge``: VICReg's variance term.

Predict before you run: latents that lie on one line through the origin, with
a large spread along it. Which meter sees the collapse, and does the variance
hinge? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_meters.py
    uv run pytest tests/test_solutions.py -k meters -v
"""

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
    # TODO(you): two lines (F.normalize, then .std(0)).
    ...


def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy) of the covariance eigenvalues, normalised to sum to 1."""
    zc = z - z.mean(0)
    ev = torch.linalg.eigvalsh(zc.T @ zc / (len(z) - 1)).clamp_min(0)
    # TODO(you): normalise ev to sum to 1, drop the zeros, return exp(-sum q log q).
    ...


def variance_hinge(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg's variance term: mean over dims of max(0, gamma - std), std over the batch."""
    # TODO(you): one line; use sqrt(var + 1e-4) for the std, as VICReg does.
    ...


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
