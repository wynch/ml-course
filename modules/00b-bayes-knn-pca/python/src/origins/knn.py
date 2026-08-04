"""k-nearest neighbours, and the bound that makes it respectable.

The classifier is four lines of numpy. The interesting part is
:func:`cover_hart_bound`: Cover & Hart (1967) proved that as the training set
grows, the 1-NN error rate settles at a value no worse than *twice* the Bayes
error. Half the information is in the labels of the nearest points; the other
half you cannot have.

Everything here is written for the two-Gaussian family in :mod:`origins.data`,
where the Bayes error is available in closed form and the asymptotic 1-NN error
can be integrated numerically — so the bound is not asserted, it is measured.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import norm


class KNN:
    """Brute-force k-NN. Fitting is remembering; all the work is at query time.

    ``weights='uniform'`` is the plain majority vote; ``weights='distance'``
    weights each neighbour by ``1/(d + eps)``.
    """

    def __init__(self, k: int = 5, weights: str = "uniform"):
        self.k = int(k)
        self.weights = weights
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def _neighbours(self, X: np.ndarray, chunk: int = 512):
        """Yield (distances, label-indices) of the k nearest, chunked by query."""
        X = np.asarray(X, dtype=np.float64)
        sq_train = (self.X_ ** 2).sum(axis=1)
        for start in range(0, len(X), chunk):
            Q = X[start : start + chunk]
            # ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2 — one matmul, no python loop
            d2 = (Q ** 2).sum(axis=1)[:, None] - 2.0 * Q @ self.X_.T + sq_train[None, :]
            np.maximum(d2, 0.0, out=d2)
            idx = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]
            take = np.take_along_axis(d2, idx, axis=1)
            order = np.argsort(take, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            yield np.sqrt(np.take_along_axis(take, order, axis=1)), idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = []
        for dist, idx in self._neighbours(X):
            labels = self.y_[idx]                       # (chunk, k)
            w = np.ones_like(dist) if self.weights == "uniform" else 1.0 / (dist + 1e-12)
            votes = np.stack(
                [np.where(labels == c, w, 0.0).sum(axis=1) for c in self.classes_], axis=1
            )
            out.append(self.classes_[votes.argmax(axis=1)])
        return np.concatenate(out)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())


# ─────────────────────── the Cover-Hart machinery ───────────────────────


def bayes_error_two_gaussians(sep: float, sigma: float = 1.0) -> float:
    """Exact Bayes error for two equal-prior spherical Gaussians.

    With equal covariance ``sigma^2 I`` the optimal boundary is the perpendicular
    bisector of the two means, and the error is the tail mass on the wrong side:
    ``Phi(-sep / (2*sigma))``. Note it depends only on the ratio — the
    Mahalanobis distance between the classes.
    """
    return float(norm.cdf(-sep / (2.0 * sigma)))


def eta_two_gaussians(X: np.ndarray, sep: float, sigma: float = 1.0) -> np.ndarray:
    """P(y=1 | x) for the same family — a logistic in the first coordinate.

    Taking the log of the ratio of the two Gaussian densities, everything
    quadratic cancels (the covariances are equal) and what is left is linear:
    ``log odds = sep * x0 / sigma^2``.
    """
    return expit(sep * np.asarray(X)[:, 0] / (sigma ** 2))


def cover_hart_bound(bayes_err: float, n_classes: int = 2) -> float:
    """The Cover-Hart ceiling ``R* (2 - M/(M-1) R*)`` on the asymptotic 1-NN error.

    For two classes this is ``2 R* (1 - R*)``, which is at most ``2 R*`` and is
    tight only when ``R*`` is tiny. It is a statement about the *limit* as the
    training set grows, not about any finite sample.
    """
    R = float(bayes_err)
    return R * (2.0 - (n_classes / (n_classes - 1.0)) * R)


def asymptotic_1nn_error(eta: np.ndarray, n_classes: int = 2) -> float:
    """Monte-Carlo estimate of the limiting 1-NN error ``E[2 eta (1 - eta)]``.

    Where the true 1-NN limit actually sits, given samples of the posterior
    ``eta(x)`` drawn from the marginal of x. As the training set grows the
    nearest neighbour's label becomes an independent draw from ``eta(x)``, so
    the two disagree with probability ``2 eta (1 - eta)``.
    """
    if n_classes != 2:
        raise NotImplementedError("closed form implemented for two classes")
    eta = np.asarray(eta, dtype=np.float64)
    return float((2.0 * eta * (1.0 - eta)).mean())
