"""PCA by power iteration, plus a `numpy.linalg.eigh` cross-check.

Power iteration is the oldest trick for eigenvectors and the easiest to believe:
multiply any starting vector by the covariance matrix over and over, renormalise
each time, and the component along the top eigenvector grows fastest, so what
survives *is* the top eigenvector. Strip it out (deflation) and repeat for the
next one.

Convergence is geometric in the eigenvalue gap: the ratio ``|lambda2/lambda1|``
is how much of the runner-up survives each iteration, so nearly-equal
eigenvalues converge slowly. The scripts report the observed iteration counts.
"""

from __future__ import annotations

import numpy as np


def power_iteration(
    A: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    seed: int = 0,
    trace: bool = False,
):
    """Top eigenpair of a symmetric matrix ``A``.

    Returns ``(eigenvalue, eigenvector, n_iter)``, or
    ``(eigenvalue, eigenvector, n_iter, trace_list)`` when ``trace=True``, where
    the trace holds the unit vector after each iteration (used by the explorable
    and the figures to animate convergence).

    Stops when the vector stops moving: ``||v_new - v_old|| < tol``, with the
    sign of ``v_new`` aligned to ``v_old`` first so a sign flip is not mistaken
    for movement.
    """
    A = np.asarray(A, dtype=np.float64)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    hist = [v.copy()]
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        w = A @ v
        nrm = np.linalg.norm(w)
        if nrm < 1e-300:            # A annihilates v: eigenvalue 0
            return 0.0, v, n_iter
        w /= nrm
        if np.dot(w, v) < 0:        # align signs before measuring movement
            w = -w
        delta = np.linalg.norm(w - v)
        v = w
        hist.append(v.copy())
        if delta < tol:
            break
    lam = float(v @ A @ v)          # Rayleigh quotient
    if trace:
        return lam, v, n_iter, hist
    return lam, v, n_iter


def top_eigenpairs(A: np.ndarray, k: int, *, tol: float = 1e-12, seed: int = 0):
    """The top ``k`` eigenpairs by power iteration **with deflation**.

    After finding ``(lam, v)`` we subtract ``lam * v v^T`` from ``A``. That
    leaves every other eigenpair untouched and drives this one to zero, so the
    next power iteration finds the runner-up. Errors accumulate across
    deflations, which is exactly why production code uses a QR-based routine —
    compare the agreement with ``eigh`` as ``k`` grows.
    """
    A = np.array(A, dtype=np.float64, copy=True)
    vals, vecs, iters = [], [], []
    for j in range(k):
        lam, v, n = power_iteration(A, tol=tol, seed=seed + j)
        vals.append(lam)
        vecs.append(v)
        iters.append(n)
        A -= lam * np.outer(v, v)
    return np.array(vals), np.array(vecs).T, iters


class PCA:
    """Covariance PCA. ``solver='power'`` (ours) or ``solver='eigh'`` (numpy).

    Both compute eigenvectors of the same covariance matrix
    ``C = X_c^T X_c / (n - 1)``; the point of keeping both is that the module
    can print their disagreement instead of asking you to trust one.
    """

    def __init__(self, n_components: int, *, solver: str = "power", tol: float = 1e-10, seed: int = 0):
        self.n_components = int(n_components)
        self.solver = solver
        self.tol = tol
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None       # (k, D), rows are PCs
        self.explained_variance_: np.ndarray | None = None
        self.total_variance_: float | None = None
        self.n_iter_: list[int] | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        C = Xc.T @ Xc / (len(X) - 1)
        self.total_variance_ = float(np.trace(C))
        if self.solver == "power":
            vals, vecs, iters = top_eigenpairs(C, self.n_components, tol=self.tol, seed=self.seed)
            self.n_iter_ = iters
        elif self.solver == "eigh":
            w, V = np.linalg.eigh(C)                     # ascending
            order = np.argsort(w)[::-1][: self.n_components]
            vals, vecs = w[order], V[:, order]
            self.n_iter_ = None
        else:
            raise ValueError(f"unknown solver {self.solver!r}")
        self.explained_variance_ = vals
        self.components_ = _fix_signs(vecs.T)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=np.float64) @ self.components_ + self.mean_

    def reconstruct(self, X: np.ndarray, k: int | None = None) -> np.ndarray:
        """Project onto the first ``k`` components and come straight back."""
        k = self.n_components if k is None else k
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        W = self.components_[:k]
        return Xc @ W.T @ W + self.mean_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.explained_variance_ / self.total_variance_


def _fix_signs(components: np.ndarray) -> np.ndarray:
    """Make each component's largest-magnitude entry positive.

    An eigenvector is only defined up to sign, so power iteration and ``eigh``
    routinely return opposite ones. Every comparison in this module fixes the
    sign first; without this the "disagreement" between solvers is meaningless.
    """
    out = np.array(components, dtype=np.float64, copy=True)
    for i, row in enumerate(out):
        if row[np.argmax(np.abs(row))] < 0:
            out[i] = -row
    return out


def reconstruction_error(X: np.ndarray, pca: "PCA", k: int) -> float:
    """Mean squared reconstruction error per feature, using ``k`` components."""
    Xr = pca.reconstruct(X, k=k)
    return float(((np.asarray(X, dtype=np.float64) - Xr) ** 2).mean())
