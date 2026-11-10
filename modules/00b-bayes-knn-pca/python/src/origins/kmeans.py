"""k-means two ways: Lloyd's full pass, and a moving average over mini-batches.

Both rules share one step, :func:`assign`: each point goes to its nearest
centre. That is Part 3's 1-NN query with the centres as the stored set (the
tests check it against :class:`origins.knn.KNN` with ``k=1``). They differ only
in how the centres move afterwards.

* :func:`lloyd` sets every centre to the mean of all its points, once per pass.
* :func:`moving_average_kmeans` keeps a running count ``N_j`` and a running sum
  ``S_j`` per centre, updates both from each mini-batch with a step ``alpha``,
  and reads the centre off as ``S_j / N_j``. The step decays on a cosine from
  ``alpha0`` to ``alpha0 / 10``, and one exact pass at the end sets each centre
  to the mean of its points. That is the core of e-means (Bessa et al., 2026,
  arXiv:2611.09501, Algorithm 1 and eq. 2-5).

Left out on purpose: the paper's two fitted power laws for the batch and the
step (the caller fixes both), and its revival of dead clusters (the authors
switch it off for ``d <= 2`` themselves).

With ``batch = n``, ``alpha0 = 1`` and no decay, one moving-average pass is one
Lloyd pass; ``tests/test_origins.py`` holds the two to that.
"""

from __future__ import annotations

import numpy as np

#: The step's floor as a fraction of ``alpha0``; the paper's epsilon (eq. 5).
STEP_FLOOR = 0.1
#: Starting count per centre, so ``S / N`` returns the initial centres exactly
#: while carrying almost no weight (the paper's initialization, Section 3.1).
N_INIT = 1e-10


def assign(X: np.ndarray, C: np.ndarray):
    """Nearest centre for every row of X: ``(labels (n,), squared distances (n,))``.

    The only place in this file that computes a distance; both rules call it.
    """
    # ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2, the identity knn.py uses
    d2 = (X ** 2).sum(axis=1)[:, None] - 2.0 * X @ C.T + (C ** 2).sum(axis=1)[None, :]
    labels = d2.argmin(axis=1)
    best = np.take_along_axis(d2, labels[:, None], axis=1)[:, 0]
    return labels, np.maximum(best, 0.0)


def kmeans_error(X: np.ndarray, C: np.ndarray) -> float:
    """The k-means objective: mean squared distance from each point to its nearest centre."""
    return float(assign(X, C)[1].mean())


def batch_stats(X: np.ndarray, labels: np.ndarray, k: int):
    """Count ``n_j`` and sum ``s_j`` of the points each centre received."""
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    sums = np.zeros((k, X.shape[1]))
    np.add.at(sums, labels, X)
    return counts, sums


def centre_means(X: np.ndarray, labels: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Each centre moved to the mean of its points; a centre with none stays put."""
    counts, sums = batch_stats(X, labels, len(C))
    out = C.copy()
    hit = counts > 0
    out[hit] = sums[hit] / counts[hit, None]
    return out


def init_from_data(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k distinct data points, drawn uniformly: the start both rules share."""
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), size=k, replace=False)].copy()


def lloyd(X: np.ndarray, C0: np.ndarray, passes: int):
    """Lloyd (1957, published 1982). Returns ``(centres, error before and after each pass)``."""
    C = np.asarray(C0, dtype=np.float64).copy()
    errs = [kmeans_error(X, C)]
    for _ in range(passes):
        labels, _ = assign(X, C)
        C = centre_means(X, labels, C)
        errs.append(kmeans_error(X, C))
    return C, errs


def cosine_step(t: int, total: int, alpha0: float, floor: float = STEP_FLOOR) -> float:
    """Step for update ``t`` of ``total``: ``alpha0`` at the first, ``floor * alpha0`` at the end."""
    return alpha0 * (floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * t / total)))


def moving_average_kmeans(
    X: np.ndarray,
    C0: np.ndarray,
    passes: int,
    *,
    batch: int,
    alpha0: float,
    seed: int = 0,
    cosine: bool = True,
    finalize: bool = True,
):
    """The moving-average rule. Returns ``(centres, errors)``.

    ``errors`` holds the error at the start, after each pass over the data, and,
    if ``finalize``, after the exact pass. That pass assigns every point to the
    final centres and moves each centre to the mean of its points: one Lloyd
    step, so it cannot raise the error. (The paper reuses the labels of the last
    pass instead, which saves the pass; its guarantee is against the errors seen
    during that pass, not against the final centres.)
    """
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C0, dtype=np.float64).copy()
    n, k = len(X), len(C)
    steps_per_pass = n // batch
    total = passes * steps_per_pass
    N = np.full(k, N_INIT)
    S = N[:, None] * C
    rng = np.random.default_rng(seed)
    errs = [kmeans_error(X, C)]
    t = 0
    for _ in range(passes):
        order = rng.permutation(n)
        for b in range(steps_per_pass):
            xb = X[order[b * batch : (b + 1) * batch]]
            labels, _ = assign(xb, C)
            n_j, s_j = batch_stats(xb, labels, k)
            alpha = cosine_step(t, total, alpha0) if cosine else alpha0
            N = (1.0 - alpha) * N + alpha * n_j
            S = (1.0 - alpha) * S + alpha * s_j
            live = N > 0  # N reaches 0 only when alpha = 1 and the centre got no points
            C[live] = S[live] / N[live, None]
            t += 1
        errs.append(kmeans_error(X, C))
    if finalize:
        before = errs[-1]
        labels, _ = assign(X, C)
        C = centre_means(X, labels, C)
        errs.append(kmeans_error(X, C))
        assert errs[-1] <= before + 1e-12 * max(1.0, before), (before, errs[-1])
    return C, errs
