"""Exercise 3 — power iteration and deflation, checked against numpy.

Two `# TODO(you):` blocks:

1. `my_power_iteration` — repeatedly multiply by the matrix and renormalise.
2. `my_top_eigenpairs` — after each eigenpair, *deflate* the matrix so the next
   iteration finds the runner-up instead of the same vector again.

The check runs both on a 6x6 symmetric matrix and on the real 784x784
FashionMNIST-shaped covariance built from seeded random data, and compares every
eigenvalue and every eigenvector direction against `numpy.linalg.eigh`.

Run:  cd python && uv run ../exercises/ex3_power_iteration.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from origins.pca import _fix_signs

TOL = 1e-12
MAX_ITER = 20_000


def my_power_iteration(A, seed=0, tol=TOL, max_iter=MAX_ITER):
    """Return (eigenvalue, unit eigenvector, iterations) for the dominant pair."""
    A = np.asarray(A, dtype=np.float64)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v)

    # TODO(you): iterate until v stops moving.
    #   w = A @ v
    #   renormalise w to unit length
    #   flip w's sign if it points opposite to v (an eigenvector is only defined
    #     up to sign, and without this a -1 eigenvalue never "converges")
    #   stop when ||w - v|| < tol, remembering how many iterations it took
    #
    # Then the eigenvalue is the Rayleigh quotient  v @ A @ v.
    #
    # return lam, v, n_iter
    raise NotImplementedError("implement power iteration")


def my_top_eigenpairs(A, k, seed=0):
    """Return (values (k,), vectors (D, k) as columns) — largest first."""
    A = np.array(A, dtype=np.float64, copy=True)
    vals, vecs = [], []
    for j in range(k):
        lam, v, _ = my_power_iteration(A, seed=seed + j)
        vals.append(lam)
        vecs.append(v)
        # TODO(you): deflate. Subtract this eigenpair's contribution from A so
        # the next call finds the *next* eigenvector rather than this one again.
        # Hint: for a symmetric A with unit eigenvector v, the rank-1 piece it
        # contributes is lam * outer(v, v).
        raise NotImplementedError("deflate the matrix")
    return np.array(vals), np.array(vecs).T


def _compare(name, A, k):
    vals, vecs = my_top_eigenpairs(A, k)
    w, V = np.linalg.eigh(A)
    order = np.argsort(w)[::-1][:k]
    w, V = w[order], V[:, order]
    rel = np.abs(vals - w) / np.abs(w)
    cos = np.abs((_fix_signs(vecs.T) * _fix_signs(V.T)).sum(axis=1))
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    print(f"{name}: top {k} eigenpairs")
    print(f"  eigenvalues (yours) {np.round(vals, 6).tolist()}")
    print(f"  eigenvalues (eigh)  {np.round(w, 6).tolist()}")
    print(f"  max relative eigenvalue error {rel.max():.3e}")
    print(f"  max eigenvector angle         {ang.max():.3e} degrees")
    return rel.max() < 1e-8 and ang.max() < 1e-4


def _check():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(6, 6))
    small = B @ B.T                                   # symmetric, positive definite
    ok = _compare("6x6 random symmetric", small, 4)

    X = rng.normal(size=(2000, 784)) @ rng.normal(size=(784, 784)) * 0.05
    C = np.cov(X, rowvar=False)                       # a real 784x784 covariance
    ok = _compare("784x784 covariance", C, 5) and ok

    print("\n" + ("PASS — power iteration + deflation agrees with LAPACK."
                  if ok else "FAIL — implement the two TODO blocks above."))
    return ok


if __name__ == "__main__":
    _check()
