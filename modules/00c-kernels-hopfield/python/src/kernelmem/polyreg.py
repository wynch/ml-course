"""Minimum-norm polynomial regression — the vehicle for double descent.

Fit y ≈ Φ(x)β where Φ is a Legendre basis of a chosen degree. Two regimes:

- **Underparameterised** (p = degree+1 ≤ n): the usual least-squares fit; more
  degrees means less bias and more variance — the textbook U.
- **Overparameterised** (p > n): infinitely many β interpolate the training
  data exactly. ``np.linalg.pinv`` picks the one with the smallest ‖β‖₂. That
  choice — the *implicit bias* — is what makes the overparameterised regime
  behave at all, and it is the same kind of implicit bias that gradient
  descent brings to a wildly overparameterised neural network.

Right at the boundary p = n there is exactly one interpolant and no freedom to
pick a well-behaved one, so ‖β‖ blows up. That spike is the interpolation
threshold, and the fall on the far side of it is the second descent.

Why Legendre and not raw powers x^k? Because min-norm is a statement about the
*coordinates*, so the basis is part of the model. The raw monomial basis is
catastrophically ill-conditioned past degree ~15 and you would be measuring
float error, not statistics. Legendre polynomials are orthogonal on [−1, 1],
which keeps the design matrix sane all the way to degree 80.
"""

from __future__ import annotations

import numpy as np


def legendre_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Design matrix with columns √((2k+1)/2)·P_k(x), k = 0 … degree.

    The scale factor makes the columns orthonormal in the L²([−1,1]) sense, so
    "small ‖β‖" means the same thing at every degree.
    """
    x = np.asarray(x, dtype=float)
    V = np.polynomial.legendre.legvander(x, degree)
    k = np.arange(degree + 1)
    return V * np.sqrt((2.0 * k + 1.0) / 2.0)


def fit_min_norm(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """β = Φ⁺y — least squares when tall, minimum-norm interpolation when wide."""
    return np.linalg.pinv(legendre_features(x, degree)) @ np.asarray(y, dtype=float)


def predict(beta: np.ndarray, x: np.ndarray, degree: int) -> np.ndarray:
    return legendre_features(x, degree) @ beta


def truth(x: np.ndarray) -> np.ndarray:
    """The function being learned: f(x) = sin(3x) + 0.35x on [−1, 1]."""
    x = np.asarray(x, dtype=float)
    return np.sin(3.0 * x) + 0.35 * x


N_TRAIN = 20
SIGMA = 0.15
DATA_SEED = 20250803
N_TEST = 200


def make_dataset(n_train: int = N_TRAIN, sigma: float = SIGMA, seed: int = DATA_SEED,
                 n_test: int = N_TEST):
    """The one dataset the module and the explorable both use.

    Training x is uniform on [−1, 1] (sorted, for tidy plotting); test x is an
    evenly spaced grid and its y is the *noise-free* truth, so test error
    measures the fit to the signal rather than to a second sample of noise.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, n_train))
    y = truth(x) + sigma * rng.standard_normal(n_train)
    xt = np.linspace(-1.0, 1.0, n_test)
    return x, y, xt, truth(xt)


def sweep(x, y, xt, yt, degrees) -> dict:
    """Train and test MSE for each degree, on one fixed dataset."""
    degrees = list(degrees)
    train, test, norms = [], [], []
    for d in degrees:
        beta = fit_min_norm(x, y, d)
        train.append(float(np.mean((predict(beta, x, d) - y) ** 2)))
        test.append(float(np.mean((predict(beta, xt, d) - yt) ** 2)))
        norms.append(float(np.linalg.norm(beta)))
    return {"degrees": degrees, "train": train, "test": test, "beta_norm": norms}


def sweep_averaged(degrees, n_train: int = N_TRAIN, sigma: float = SIGMA,
                   trials: int = 200, seed: int = 0, n_test: int = 500) -> dict:
    """Median train/test MSE over many fresh draws of the training set.

    The median rather than the mean: near the interpolation threshold a single
    unlucky draw produces an error of 1e13 and drags any average with it.
    """
    degrees = list(degrees)
    rng = np.random.default_rng(seed)
    xt = np.linspace(-1.0, 1.0, n_test)
    yt = truth(xt)
    tr = np.zeros((len(degrees), trials))
    te = np.zeros((len(degrees), trials))
    for t in range(trials):
        x = rng.uniform(-1.0, 1.0, n_train)
        y = truth(x) + sigma * rng.standard_normal(n_train)
        for i, d in enumerate(degrees):
            beta = fit_min_norm(x, y, d)
            tr[i, t] = np.mean((predict(beta, x, d) - y) ** 2)
            te[i, t] = np.mean((predict(beta, xt, d) - yt) ** 2)
    return {
        "degrees": degrees,
        "train": np.median(tr, axis=1).tolist(),
        "test": np.median(te, axis=1).tolist(),
    }


def bias_variance(degrees, n_train: int = N_TRAIN, sigma: float = SIGMA,
                  trials: int = 300, seed: int = 5, n_test: int = 200) -> dict:
    """Decompose test error into bias², variance and the noise floor.

    Over ``trials`` independent training sets, for each degree:
        bias²(x) = (E[f̂(x)] − f(x))² ,   var(x) = E[(f̂(x) − E[f̂(x)])²].
    Only run this in the underparameterised regime — past the threshold the
    variance is astronomically large and the decomposition stops being legible.
    """
    degrees = list(degrees)
    rng = np.random.default_rng(seed)
    xt = np.linspace(-1.0, 1.0, n_test)
    yt = truth(xt)
    bias2, var = [], []
    for d in degrees:
        preds = np.zeros((trials, n_test))
        for t in range(trials):
            x = rng.uniform(-1.0, 1.0, n_train)
            y = truth(x) + sigma * rng.standard_normal(n_train)
            preds[t] = predict(fit_min_norm(x, y, d), xt, d)
        mean_pred = preds.mean(0)
        bias2.append(float(np.mean((mean_pred - yt) ** 2)))
        var.append(float(np.mean(preds.var(0))))
    return {"degrees": degrees, "bias2": bias2, "variance": var,
            "noise": float(sigma**2)}
