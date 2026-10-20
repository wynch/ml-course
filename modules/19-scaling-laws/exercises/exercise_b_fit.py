"""Exercise (b): fit a power law, and check the fitter on data with a known answer first.

A power law L = exp(a) * N^(-alpha) is a straight line in log-log space:
log L = a - alpha log N. So fit a line to (log N, log L) by least squares.
Before you trust it on real losses, feed it losses made from alpha = 0.1 and
see whether it gives 0.1 back.

Predict before you run: with noise of SD 0.01 on log L, six sizes spanning a
factor 62 in N, how far from 0.1 will the fitted alpha typically land?
0.0003, 0.003 or 0.03?

    uv run python ../exercises/exercise_b_fit.py
    uv run pytest tests/test_solutions.py -k fit -v
"""

from __future__ import annotations

import numpy as np

N = np.array([50_880, 113_184, 200_064, 447_552, 1_237_440, 3_159_552], dtype=float)


def fit_power(x, y) -> tuple[float, float, float]:
    """Fit y = exp(a) * x^(-alpha) by least squares on log y against log x.
    Returns (alpha, a, R^2), R^2 measured in log space."""
    # TODO(you): take logs, np.polyfit(..., 1), alpha is minus the slope;
    # R^2 = 1 - (sum of squared residuals) / (sum of squared deviations of log y from its mean).
    raise NotImplementedError("TODO(you): write fit_power first")


def synthetic(alpha: float, noise: float, seed: int) -> np.ndarray:
    """Losses that follow log L = log 2 - alpha log N exactly, plus Gaussian noise on log L."""
    # TODO(you): np.random.default_rng(seed).normal(0, noise, len(N)) added to log L, then exp.
    raise NotImplementedError("TODO(you): write synthetic first")


def main() -> None:
    for noise in (0.0, 0.01, 0.05):
        alphas = [fit_power(N, synthetic(0.1, noise, s))[0] for s in range(200)]
        print(f"noise {noise:.2f}: alpha recovered {np.mean(alphas):.4f} +- {np.std(alphas):.4f}")


if __name__ == "__main__":
    main()
