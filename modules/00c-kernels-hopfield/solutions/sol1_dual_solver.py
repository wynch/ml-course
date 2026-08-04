"""Solution to exercise 1 — the dual SVM's gradient and feasible projection.

The dual problem is

    maximise  W(α) = Σαᵢ − ½ αᵀQα      with Q = (yyᵀ) ⊙ K
    subject to  αᵢ ≥ 0   and   αᵀy = 0.

Projected gradient ascent needs exactly two things you have to supply:

  1. ``dual_gradient`` — ∇W(α).
  2. ``project_feasible`` — the Euclidean projection back onto the feasible set.
     Hint: the KKT conditions say the answer is ``clip(v − μy, 0, ∞)`` for one
     scalar μ, and αᵀy is a *non-increasing* function of μ, so bisect on μ.

The checks at the bottom verify the KKT conditions of the fitted SVM — support
vectors exactly on the margin, nothing inside it, and ‖w‖² = Σαᵢ, which only
holds at the true optimum.

Run:  cd python && uv run ../solutions/sol1_dual_solver.py
"""

import numpy as np


def dual_gradient(alpha, Q):
    """∇W(α) for W(α) = Σαᵢ − ½ αᵀQα."""
    return 1.0 - Q @ alpha


def project_feasible(v, y):
    """Euclidean projection of v onto {α : α ≥ 0, αᵀy = 0}.

    Return ``np.clip(v - mu * y, 0, None)`` for the μ that makes the result
    satisfy αᵀy = 0. Find μ by bisection: g(μ) = clip(v − μy, 0, ∞)ᵀy starts
    positive for very negative μ and ends negative for very positive μ.
    """
    def g(mu):
        return float(np.clip(v - mu * y, 0.0, None) @ y)

    span = max(1.0, float(np.max(np.abs(v))))
    lo, hi = -span, span
    while g(lo) < 0.0:      # g is non-increasing: push the bracket outwards
        lo *= 2.0
    while g(hi) > 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return np.clip(v - 0.5 * (lo + hi) * y, 0.0, None)


# ─────────────────────────────────────────── everything below is provided ──

def fit(X, y, iters=4000):
    K = X @ X.T
    Q = np.outer(y, y) * K
    lr = 1.0 / max(float(np.max(np.linalg.eigvalsh(Q))), 1e-12)
    alpha = np.zeros(len(y))
    for _ in range(iters):
        alpha = project_feasible(alpha + lr * dual_gradient(alpha, Q), y)
    w = (alpha * y) @ X
    sv = alpha > 1e-5 * max(alpha.max(), 1e-12)
    b = float(np.mean(y[sv] - (alpha * y) @ K[:, sv]))
    return alpha, w, b, sv


def blobs(n=24, seed=12, gap=2.4):
    rng = np.random.default_rng(seed)
    half = n // 2
    a = rng.normal(loc=(-gap / 2, -gap / 3), scale=0.62, size=(half, 2))
    b = rng.normal(loc=(+gap / 2, +gap / 3), scale=0.62, size=(n - half, 2))
    return np.vstack([a, b]), np.concatenate([-np.ones(half), np.ones(n - half)])


def main():
    X, y = blobs()
    alpha, w, b, sv = fit(X, y)
    f = y * (X @ w + b)

    print(f"support vectors: {int(sv.sum())} of {len(y)}")
    print(f"w = [{w[0]:.6f}, {w[1]:.6f}]  b = {b:.6f}")
    print(f"margin 2/||w||      = {2 / np.linalg.norm(w):.6f}")
    print(f"margin 2/sqrt(sum a)= {2 / np.sqrt(alpha.sum()):.6f}")

    checks = [
        ("alpha >= 0", bool((alpha >= -1e-9).all())),
        ("sum(alpha*y) == 0", abs(float(alpha @ y)) < 1e-8),
        ("support vectors sit on the margin", bool(np.allclose(f[sv], 1.0, atol=1e-4))),
        ("no point inside the margin", bool((f >= 1 - 1e-4).all())),
        ("||w||^2 == sum(alpha)", abs(float(w @ w) - float(alpha.sum())) < 1e-6),
        ("exactly 3 support vectors", int(sv.sum()) == 3),
        ("margin is 1.4627", abs(2 / np.linalg.norm(w) - 1.462661) < 1e-4),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\nall good" if all(ok for _n, ok in checks) else "\nnot yet")


if __name__ == "__main__":
    main()
