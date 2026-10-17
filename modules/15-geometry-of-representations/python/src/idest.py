"""Two intrinsic-dimension estimators, TwoNN and MLE, and a known-dimension test set.

Both read only the distances from each point to its nearest neighbours.

- TwoNN (Facco et al., 2017): with r1, r2 the distances to the first and second
  neighbour, mu = r2 / r1 follows P(mu > m) = m^(-d) on a d-dimensional set of
  locally even density. Sort the mu, drop the largest 10%, and fit
  -log(1 - F(mu)) = d log(mu) through the origin.
- MLE (Levina and Bickel, 2005): with T_1 <= ... <= T_k the distances to the k
  nearest neighbours, 1/d is estimated by the mean of log(T_k / T_j), j < k.
  We average that inverse over all points (MacKay and Ghahramani, 2005), then
  average d over k = 10..20.
"""

from __future__ import annotations

import numpy as np

K_RANGE = range(10, 21)


def knn_dists(x: np.ndarray, k: int, chunk: int = 500) -> np.ndarray:
    """Distances to the k nearest other points, sorted, float64, exact. (n, k)."""
    import torch

    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    sq = (xt * xt).sum(1)
    out = np.empty((len(x), k))
    for s in range(0, len(x), chunk):
        xb = xt[s:s + chunk]
        d2 = sq[s:s + chunk, None] + sq[None, :] - 2 * xb @ xt.T
        d2[torch.arange(len(xb)), torch.arange(s, s + len(xb))] = float("inf")   # not its own neighbour
        v, _ = torch.topk(d2, k, dim=1, largest=False)
        out[s:s + chunk] = v.clamp_min(0).sqrt().numpy()
    return out


def twonn(d: np.ndarray, discard: float = 0.1) -> float:
    """TwoNN from the neighbour distances d (n, >= 2)."""
    mu = np.sort(d[:, 1] / d[:, 0])
    n = len(mu)
    keep = int(np.floor(n * (1 - discard)))
    xs = np.log(mu[:keep])
    ys = -np.log(1 - np.arange(1, keep + 1) / n)
    return float((xs @ ys) / (xs @ xs))


def mle_k(d: np.ndarray, k: int) -> float:
    """Levina-Bickel MLE at one k, averaged over points as MacKay and Ghahramani."""
    logs = np.log(d[:, k - 1:k] / d[:, : k - 1])       # log(T_k / T_j), j = 1..k-1
    return float(1.0 / logs.mean())


def mle(d: np.ndarray) -> float:
    return float(np.mean([mle_k(d, k) for k in K_RANGE]))


def estimate(x: np.ndarray) -> dict:
    """TwoNN and MLE (k 10-20) on the rows of x, after dropping exact duplicate rows."""
    u = np.unique(x, axis=0)
    d = knn_dists(u, max(K_RANGE))
    return {"n": len(u), "duplicates_removed": len(x) - len(u), "twonn": round(twonn(d), 3),
            "mle": round(mle(d), 3), "mle_per_k": {str(k): round(mle_k(d, k), 3) for k in K_RANGE}}


def curved_cube(dim: int, n: int, ambient: int = 784, seed: int = 0) -> np.ndarray:
    """n points of a d-dimensional set of known dimension, bent into `ambient` dimensions.

    z is uniform on the cube [0, 1]^d; x = sin(z W + b) with W ~ N(0, 1) and b ~ U[0, 2 pi).
    The map is smooth, so the set has dimension d, yet it is curved: no flat subspace
    much smaller than `ambient` holds it.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1, size=(n, dim))
    w = rng.normal(0, 1, size=(dim, ambient))
    b = rng.uniform(0, 2 * np.pi, size=ambient)
    return np.sin(z @ w + b)
