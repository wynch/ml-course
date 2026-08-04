"""Least squares two ways: solve it, or walk downhill to it.

The perceptron only asks "which side?". Least squares asks "how far off?" and
in doing so introduces the object the whole course runs on: a differentiable
loss with a gradient.

    L(w) = ‖Xw − y‖² / n

Setting ∇L = 0 gives the **normal equations** ``XᵀX w = Xᵀy`` — one linear
solve, exact, no iterations. Gradient descent instead takes small steps
``w ← w − lr·∇L`` and creeps toward the same point. Comparing the two is the
cheapest possible demonstration that "training" is just a slow way to solve an
equation you sometimes cannot solve directly.
"""

from __future__ import annotations

import numpy as np


def loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Mean squared error ``‖Xw − y‖² / n``."""
    r = X @ w - y
    return float(r @ r / len(y))


def grad(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``∇L = 2·Xᵀ(Xw − y) / n``."""
    return 2.0 * X.T @ (X @ w - y) / len(y)


def normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ``XᵀX w = Xᵀy`` directly. Exact, one shot, no learning rate."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def gradient_descent(
    X: np.ndarray, y: np.ndarray, lr: float = 0.05, steps: int = 400,
    w0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain full-batch GD. Returns ``(w_final, path, losses)``.

    ``path`` has shape ``(steps + 1, d)`` so it can be drawn on a contour map.
    """
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    path = [w.copy()]
    losses = [loss(X, y, w)]
    for _ in range(steps):
        w = w - lr * grad(X, y, w)
        path.append(w.copy())
        losses.append(loss(X, y, w))
    return w, np.asarray(path), np.asarray(losses)


def stability_edge(X: np.ndarray) -> float:
    """Largest stable learning rate for this problem: ``2 / λ_max(2XᵀX/n)``.

    Gradient descent on a quadratic converges iff every eigenvalue λ of the
    Hessian satisfies ``|1 − lr·λ| < 1``. Above this value the iterates
    oscillate and diverge — the same edge module 01's explorable lets you fall
    off by hand.
    """
    H = 2.0 * X.T @ X / len(X)
    return float(2.0 / np.max(np.linalg.eigvalsh(H)))


def projection(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    """The geometry: ``ŷ = Xw*`` is the point of ``col(X)`` closest to ``y``.

    Returns the fit, the residual, and the orthogonality check ``Xᵀr`` — which
    is zero to machine precision precisely *because* ``w*`` solves the normal
    equations. Least squares is a right angle.
    """
    w = normal_equations(X, y)
    yhat = X @ w
    r = y - yhat
    return {
        "w": w,
        "yhat": yhat,
        "residual": r,
        "orthogonality": X.T @ r,
        "max_abs_orthogonality": float(np.max(np.abs(X.T @ r))),
        "rss": float(r @ r),
        "cos_angle": float((y @ yhat) / (np.linalg.norm(y) * np.linalg.norm(yhat))),
    }
