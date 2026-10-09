"""Solution to exercise 2 — ridge regression rescues a singular design.

`XᵀX w = Xᵀy` has a unique solution only when `XᵀX` is invertible. Add a
feature that is nearly a copy of another one and the matrix becomes almost
singular: the fit still passes through the data, but the coefficients explode
and swing wildly with the noise. Ridge regression adds `λ‖w‖²` to the loss,
which turns the solve into `(XᵀX + λI) w = Xᵀy` — always invertible for λ > 0.

Run:  cd python && uv run ../solutions/ex2_ridge.py
"""

import numpy as np


def collinear_data(n=60, eps=1e-3, seed=1958):
    """Design matrix [1, x, x + tiny noise] — the last two columns nearly agree."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    x_copy = x + rng.normal(0, eps, size=n)
    X = np.column_stack([np.ones(n), x, x_copy])
    y = 1.7 * x - 0.8 + rng.normal(0, 0.55, size=n)
    return X, y


def mse(X, y, w):
    r = X @ w - y
    return float(r @ r / len(y))


def normal_equations(X, y):
    """Solve XᵀX w = Xᵀy."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def ridge(X, y, lam):
    """Solve (XᵀX + λI) w = Xᵀy."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def condition_number(X):
    """κ(XᵀX) = λ_max / λ_min — how close to singular the matrix is."""
    ev = np.linalg.eigvalsh(X.T @ X)
    return float(ev[-1] / ev[0])


def main():
    X, y = collinear_data()
    print(f"κ(XᵀX) = {condition_number(X):.3e}   "
          "— past ~1e8 and f64 starts running out of digits")

    w = normal_equations(X, y)
    print(f"\nplain      ‖w‖ = {np.linalg.norm(w):9.3f}   "
          f"train MSE {mse(X, y, w):.5f}   w = {np.round(w, 3).tolist()}")

    for lam in (1e-6, 1e-3, 1e-1, 1.0):
        wr = ridge(X, y, lam)
        print(f"λ = {lam:<8g} ‖w‖ = {np.linalg.norm(wr):9.3f}   "
              f"train MSE {mse(X, y, wr):.5f}   w = {np.round(wr, 3).tolist()}")

    print("\nThe two collinear coefficients are huge and opposite without ridge: "
          "the model\nhas found a direction along which it can move a long way "
          "and barely change the fit.")


if __name__ == "__main__":
    main()
