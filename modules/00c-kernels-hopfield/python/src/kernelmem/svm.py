"""The support-vector machine, in the dual, from scratch.

The primal problem — "find the separating hyperplane with the widest margin" —
is a constrained quadratic program:

    minimise  ½‖w‖²        subject to   yᵢ(wᵀxᵢ + b) ≥ 1  for every i.

Its Lagrangian dual introduces one multiplier αᵢ ≥ 0 per training point and
turns into a *maximisation* that touches the data only through inner products:

    maximise  W(α) = Σᵢ αᵢ − ½ ΣᵢΣⱼ αᵢαⱼ yᵢyⱼ ⟨xᵢ, xⱼ⟩
    subject to  αᵢ ≥ 0   and   Σᵢ αᵢyᵢ = 0.

Two things fall out of that shape and they are the whole point of this module:

1. **Sparsity.** At the optimum αᵢ = 0 for every point that sits strictly
   outside the margin. Only the points *on* the margin — the support vectors —
   carry weight, so the solution is a handful of examples, not all of them.
2. **The kernel trick.** ⟨xᵢ, xⱼ⟩ is the only way the data enters. Replace it
   with any kernel K(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩ and you have fitted a hyperplane
   in φ's space without ever computing φ.

We solve the dual with **projected gradient ascent**: take a step along
∇W(α) = 1 − Qα (with Q = (yyᵀ) ⊙ K), then project back onto the feasible set
{α ≥ 0, αᵀy = 0}. The projection is exact, not a heuristic — see
``project_feasible``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ────────────────────────────────────────────────────────────────── kernels ──


def linear_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """K(a, b) = aᵀb. The plain inner product — no lift at all."""
    return A @ B.T


def poly_kernel(A: np.ndarray, B: np.ndarray, degree: int = 2, coef0: float = 0.0) -> np.ndarray:
    """K(a, b) = (aᵀb + coef0)^degree.

    With ``coef0 = 0`` and ``degree = 2`` this is the *homogeneous* quadratic
    kernel whose feature map is written out explicitly in :mod:`kernelmem.lift`.
    """
    return (A @ B.T + coef0) ** degree


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """K(a, b) = exp(−γ‖a − b‖²), the Gaussian kernel.

    Its feature map is infinite-dimensional, which is exactly why you want to
    stay in kernel form: you could never write φ(x) down.
    """
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.exp(-gamma * np.maximum(sq, 0.0))


KERNELS = {"linear": linear_kernel, "poly": poly_kernel, "rbf": rbf_kernel}


def kernel_matrix(A: np.ndarray, B: np.ndarray, kind: str = "linear", **params) -> np.ndarray:
    """Gram matrix K[i, j] = K(A[i], B[j]) for one of the named kernels."""
    if kind not in KERNELS:
        raise ValueError(f"unknown kernel {kind!r}; pick one of {sorted(KERNELS)}")
    return KERNELS[kind](A, B, **params)


# ─────────────────────────────────────────────────────────────── projection ──


def project_feasible(v: np.ndarray, y: np.ndarray, C: float | None = None) -> np.ndarray:
    """Euclidean projection of ``v`` onto {α : 0 ≤ α ≤ C, αᵀy = 0}.

    The KKT conditions of that projection say the answer has the form

        α(μ) = clip(v − μ·y, 0, C)

    for a single scalar μ chosen so that α(μ)ᵀy = 0.  Because y ∈ {−1, +1},

        d/dμ [α(μ)ᵀy] = −(number of unclipped coordinates) ≤ 0,

    so g(μ) = α(μ)ᵀy is non-increasing and a bisection finds its root exactly.
    ``C = None`` means the hard-margin case with no upper bound.
    """
    hi_bound = np.inf if C is None else float(C)

    def alpha_of(mu: float) -> np.ndarray:
        return np.clip(v - mu * y, 0.0, hi_bound)

    def g(mu: float) -> float:
        return float(alpha_of(mu) @ y)

    # bracket the root by doubling outwards from 0
    lo, hi = -1.0, 1.0
    span = max(1.0, float(np.max(np.abs(v))) if v.size else 1.0)
    lo, hi = -span, span
    for _ in range(60):
        if g(lo) >= 0.0:
            break
        lo *= 2.0
    for _ in range(60):
        if g(hi) <= 0.0:
            break
        hi *= 2.0
    for _ in range(200):  # bisection to machine precision on the bracket
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return alpha_of(0.5 * (lo + hi))


# ────────────────────────────────────────────────────────────────────── SVM ──


@dataclass
class DualSVM:
    """Hard-margin (or box-constrained) SVM solved in the dual.

    Parameters
    ----------
    kernel, params
        Kernel name from :data:`KERNELS` and its keyword arguments.
    C
        Upper bound on each αᵢ. ``None`` (the default) is the hard margin of
        the classic derivation: the data *must* be separable in feature space.
    lr, iters
        Projected-gradient step size and iteration count. ``lr=None`` picks
        1/λ_max(Q), the largest step that cannot overshoot the quadratic.
    """

    kernel: str = "linear"
    params: dict = field(default_factory=dict)
    C: float | None = None
    lr: float | None = None
    iters: int = 4000
    sv_tol: float = 1e-5

    alpha: np.ndarray = field(init=False, default=None)
    b: float = field(init=False, default=0.0)
    X: np.ndarray = field(init=False, default=None)
    y: np.ndarray = field(init=False, default=None)
    history: list = field(init=False, default_factory=list)

    # ---- fitting ----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DualSVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if not set(np.unique(y)).issubset({-1.0, 1.0}):
            raise ValueError("labels must be −1 / +1")

        K = kernel_matrix(X, X, self.kernel, **self.params)
        Q = np.outer(y, y) * K

        # step size: 1/L with L the Lipschitz constant of ∇W (= λ_max of Q)
        lr = self.lr
        if lr is None:
            lam_max = float(np.max(np.linalg.eigvalsh(Q)))
            lr = 1.0 / max(lam_max, 1e-12)

        alpha = np.zeros(len(y))
        self.history = []
        for step in range(self.iters):
            grad = 1.0 - Q @ alpha              # ∇W(α)
            alpha = project_feasible(alpha + lr * grad, y, self.C)
            if step % max(1, self.iters // 200) == 0 or step == self.iters - 1:
                self.history.append((step, self.dual_objective(alpha, Q)))

        self.alpha = alpha
        self.X, self.y = X, y

        # bias from the support vectors: yᵢ = Σⱼ αⱼyⱼK(xⱼ, xᵢ) + b on the margin
        sv = self.support_mask()
        if sv.any():
            decision_no_b = (alpha * y) @ K[:, sv]
            self.b = float(np.mean(y[sv] - decision_no_b))
        else:
            self.b = 0.0
        return self

    # ---- readouts ---------------------------------------------------------
    @staticmethod
    def dual_objective(alpha: np.ndarray, Q: np.ndarray) -> float:
        """W(α) = Σαᵢ − ½ αᵀQα, the quantity projected gradient ascent climbs."""
        return float(alpha.sum() - 0.5 * alpha @ Q @ alpha)

    def support_mask(self) -> np.ndarray:
        """Boolean mask of the support vectors (αᵢ above the sparsity tolerance)."""
        scale = max(float(np.max(self.alpha)), 1e-12)
        return self.alpha > self.sv_tol * scale

    def decision_function(self, Z: np.ndarray) -> np.ndarray:
        """f(z) = Σᵢ αᵢyᵢK(xᵢ, z) + b — note it never mentions w."""
        Kz = kernel_matrix(self.X, np.asarray(Z, dtype=float), self.kernel, **self.params)
        return (self.alpha * self.y) @ Kz + self.b

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(Z))

    def weights(self) -> np.ndarray:
        """The primal w = Σᵢ αᵢyᵢxᵢ. Only meaningful for the linear kernel."""
        if self.kernel != "linear":
            raise ValueError("w exists explicitly only for the linear kernel")
        return (self.alpha * self.y) @ self.X

    def margin(self) -> float:
        """Geometric margin width 2/‖w‖.

        At the optimum of a hard-margin dual, ‖w‖² = Σᵢ αᵢ, so this is also
        2/√(Σαᵢ) — a cross-check that needs no explicit feature space and
        therefore works for every kernel.
        """
        return 2.0 / np.sqrt(max(float(self.alpha.sum()), 1e-300))


# ──────────────────────────────────────────────────────────────────── data ──


def separable_blobs(n: int = 24, seed: int = 12, gap: float = 2.4) -> tuple[np.ndarray, np.ndarray]:
    """Two linearly separable Gaussian clouds in 2D, deterministically seeded."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = rng.normal(loc=(-gap / 2, -gap / 3), scale=0.62, size=(half, 2))
    b = rng.normal(loc=(+gap / 2, +gap / 3), scale=0.62, size=(n - half, 2))
    X = np.vstack([a, b])
    y = np.concatenate([-np.ones(half), np.ones(n - half)])
    return X, y


def circles(n: int = 120, seed: int = 3, r_inner: float = 0.9, r_outer: float = 2.1,
            noise: float = 0.16) -> tuple[np.ndarray, np.ndarray]:
    """Concentric rings: the canonical set that no straight line can split."""
    rng = np.random.default_rng(seed)
    half = n // 2
    t_in = rng.uniform(0, 2 * np.pi, half)
    t_out = rng.uniform(0, 2 * np.pi, n - half)
    r_in = r_inner + noise * rng.standard_normal(half)
    r_out = r_outer + noise * rng.standard_normal(n - half)
    X = np.vstack([
        np.column_stack([r_in * np.cos(t_in), r_in * np.sin(t_in)]),
        np.column_stack([r_out * np.cos(t_out), r_out * np.sin(t_out)]),
    ])
    y = np.concatenate([np.ones(half), -np.ones(n - half)])
    return X, y
