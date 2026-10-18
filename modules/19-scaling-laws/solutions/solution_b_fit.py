"""Reference solution, exercise (b): fit a power law, and check the fitter on known data first."""

from __future__ import annotations

import numpy as np

N = np.array([50_880, 113_184, 200_064, 447_552, 1_237_440, 3_159_552], dtype=float)


def fit_power(x, y) -> tuple[float, float, float]:
    """Fit y = exp(a) * x^(-alpha) by least squares on log y against log x.
    Returns (alpha, a, R^2), R^2 measured in log space."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, a = np.polyfit(lx, ly, 1)
    pred = a + slope * lx
    r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(-slope), float(a), float(r2)


def synthetic(alpha: float, noise: float, seed: int) -> np.ndarray:
    """Losses that follow log L = log 2 - alpha log N exactly, plus Gaussian noise on log L."""
    rng = np.random.default_rng(seed)
    return np.exp(np.log(2.0) - alpha * np.log(N) + rng.normal(0, noise, len(N)))


def main() -> None:
    for noise in (0.0, 0.01, 0.05):
        alphas = [fit_power(N, synthetic(0.1, noise, s))[0] for s in range(200)]
        print(f"noise {noise:.2f}: alpha recovered {np.mean(alphas):.4f} +- {np.std(alphas):.4f}")


if __name__ == "__main__":
    main()
