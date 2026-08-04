"""Rosenblatt's perceptron, its mistake counter, and Novikoff's bound.

The learning rule is one line — on a mistake, ``w += y·x`` — and the theorem
about it is the first real guarantee in machine learning:

    a perceptron cycling over linearly separable data makes at most
    ``(R / γ)²`` mistakes before it stops making any,

where ``R = max‖x‖`` is the radius of the data and ``γ`` is the largest margin
any unit-norm separator achieves. Both quantities live in the *augmented*
space (a constant 1 appended to every point) because that is the space the
algorithm actually works in — the bias is just another weight.

Computing γ honestly is the interesting part. For separators through the
origin,

    γ = min over the convex hull of {yᵢ·xᵢ} of ‖z‖,

i.e. the distance from the origin to that hull. :func:`max_margin` finds it
with Frank–Wolfe / Gilbert's algorithm — twenty lines, no solver library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PerceptronRun:
    """Everything one training run produced."""

    w: np.ndarray                      #: final augmented weight vector [w1, w2, b]
    mistakes: int                      #: total updates performed
    epochs: int                        #: passes completed (converged or capped)
    converged: bool                    #: True if a full pass made no mistake
    snapshots: list[np.ndarray] = field(default_factory=list)
    #: (mistake index, example index) of every update, in order
    updates: list[tuple[int, int]] = field(default_factory=list)
    #: mistakes made in each epoch
    per_epoch: list[int] = field(default_factory=list)
    #: cumulative mistakes after each example processed (length = epochs·n + 1)
    trace: list[int] = field(default_factory=list)

    def predict(self, Xa: np.ndarray) -> np.ndarray:
        return np.where(Xa @ self.w > 0, 1, -1)

    def accuracy(self, Xa: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(Xa) == y).mean())


def train(
    Xa: np.ndarray,
    y: np.ndarray,
    max_epochs: int = 100,
    w0: np.ndarray | None = None,
) -> PerceptronRun:
    """Cycle over the examples in order, updating on every mistake.

    ``Xa`` must already be augmented (see :func:`perceptron.data.augment`).
    Starting from ``w = 0`` and taking a unit step keeps the run deterministic:
    no learning rate, no shuffling, no randomness anywhere.
    """
    n, d = Xa.shape
    w = np.zeros(d) if w0 is None else np.array(w0, dtype=float)
    run = PerceptronRun(w=w, mistakes=0, epochs=0, converged=False)
    run.snapshots.append(w.copy())
    run.trace.append(0)
    for epoch in range(max_epochs):
        epoch_mistakes = 0
        for i in range(n):
            if y[i] * (Xa[i] @ w) <= 0:          # mistake (ties count as wrong)
                w += y[i] * Xa[i]                 # the entire learning rule
                run.mistakes += 1
                epoch_mistakes += 1
                run.snapshots.append(w.copy())
                run.updates.append((run.mistakes, i))
            run.trace.append(run.mistakes)
        run.per_epoch.append(epoch_mistakes)
        run.epochs = epoch + 1
        if epoch_mistakes == 0:
            run.converged = True
            break
    run.w = w
    return run


def radius(Xa: np.ndarray) -> float:
    """``R = max‖x‖`` over the (augmented) data."""
    return float(np.max(np.linalg.norm(Xa, axis=1)))


def max_margin(Xa: np.ndarray, y: np.ndarray, iters: int = 20000,
               tol: float = 1e-12) -> tuple[float, np.ndarray]:
    """Largest margin achievable by a unit-norm separator through the origin.

    Frank–Wolfe on ``min ‖Σ αᵢ zᵢ‖`` over the simplex, with ``zᵢ = yᵢ·xᵢ``.
    Each step moves toward the hull vertex with the smallest projection onto
    the current point and uses the exact line-search step size. Returns
    ``(γ, u)`` where ``u`` is the unit normal of the max-margin separator.
    """
    Z = Xa * y[:, None]
    p = Z[int(np.argmin((Z * Z).sum(axis=1)))].copy()   # start at the shortest zᵢ
    for _ in range(iters):
        scores = Z @ p
        j = int(np.argmin(scores))
        d = Z[j] - p                                     # Frank–Wolfe direction
        dd = float(d @ d)
        if dd < tol:
            break
        step = float(np.clip(-(p @ d) / dd, 0.0, 1.0))   # exact line search
        if step <= 0.0:
            break
        p = p + step * d
    gamma = float(np.linalg.norm(p))
    u = p / gamma if gamma > 0 else p
    return gamma, u


def novikoff_bound(Xa: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """``R``, ``γ`` and the mistake bound ``(R/γ)²`` for this dataset.

    Frank–Wolfe brackets the true maximum margin γ\\*: the hull point it lands
    on has norm ``γ_hull ≥ γ*``, while the margin its direction actually
    achieves is ``γ ≤ γ*``. We quote the *achieved* margin, because Novikoff's
    theorem holds for any unit separator and the achieved value is the one we
    can hand you a vector for — so the bound below is honest rather than
    optimistic. ``gap`` says how tight the bracket is.
    """
    R = radius(Xa)
    gamma_hull, u = max_margin(Xa, y)
    achieved = float(np.min(y * (Xa @ u)))
    return {
        "R": R,
        "gamma": achieved,
        "gamma_hull": gamma_hull,
        "gap": gamma_hull - achieved,
        "bound": (R / achieved) ** 2,
        "u": u,
    }


def line_from_weights(w: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """y-coordinates of the decision line ``w1·x + w2·y + b = 0`` at ``xs``.

    Returns NaNs for a vertical (or undefined) boundary so plotting degrades
    gracefully — the very first update often gives ``w2 = 0``.
    """
    if abs(w[1]) < 1e-12:
        return np.full_like(xs, np.nan, dtype=float)
    return -(w[0] * xs + w[2]) / w[1]
