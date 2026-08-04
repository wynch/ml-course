"""Solution to exercise 3 — minimum-norm fits and the interpolation threshold.

The peak sits at degree n − 1, i.e. exactly where the parameter count p = degree
+ 1 equals the number of training points. Change ``N_TRAIN`` and it moves with n.

Run:  cd python && uv run ../solutions/sol3_double_descent.py
"""

import numpy as np

N_TRAIN = 20      # try 12, 20, 30 — the peak follows
SIGMA = 0.15
SEED = 20250803
DEGREES = list(range(1, 41))


def truth(x):
    return np.sin(3.0 * x) + 0.35 * x


def legendre_features(x, degree):
    """Orthonormalised Legendre basis, columns k = 0 … degree."""
    V = np.polynomial.legendre.legvander(np.asarray(x, dtype=float), degree)
    k = np.arange(degree + 1)
    return V * np.sqrt((2.0 * k + 1.0) / 2.0)


def fit_min_norm(x, y, degree):
    """Return β. Tall design matrix → least squares; wide → minimum-norm.

    Hint: one numpy function covers both cases and is named after the
    Moore–Penrose pseudo-inverse.
    """
    return np.linalg.pinv(legendre_features(x, degree)) @ np.asarray(y, dtype=float)


def peak_degree(x, y, xt, yt, degrees):
    """The degree whose test MSE is largest — the interpolation threshold."""
    test = []
    for d in degrees:
        beta = fit_min_norm(x, y, d)
        test.append(float(np.mean((legendre_features(xt, d) @ beta - yt) ** 2)))
    return degrees[int(np.argmax(test))], test


# ─────────────────────────────────────────── everything below is provided ──

def make_dataset(n_train=N_TRAIN, sigma=SIGMA, seed=SEED, n_test=200):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, n_train))
    y = truth(x) + sigma * rng.standard_normal(n_train)
    xt = np.linspace(-1.0, 1.0, n_test)
    return x, y, xt, truth(xt)


def main():
    x, y, xt, yt = make_dataset()
    n = len(x)
    peak, test = peak_degree(x, y, xt, yt, DEGREES)

    print(f"n = {n} training points")
    print(f"worst test MSE at degree {peak} (p = {peak + 1} parameters)")
    print()
    print(" degree   params   test MSE      train MSE")
    for d in [1, 3, 6, 10, n - 2, n - 1, n, n + 5, 30, 40]:
        beta = fit_min_norm(x, y, d)
        tr = float(np.mean((legendre_features(x, d) @ beta - y) ** 2))
        te = test[DEGREES.index(d)]
        mark = "  ← p = n" if d == n - 1 else ""
        print(f"  {d:5d}   {d + 1:6d}   {te:11.4e}   {tr:11.4e}{mark}")

    best_before = int(np.argmin(test[: n - 2]))
    checks = [
        ("the peak lands on p = n (within one degree)", abs(peak - (n - 1)) <= 1),
        ("training error is ~0 past the threshold",
         float(np.mean((legendre_features(x, n + 5) @ fit_min_norm(x, y, n + 5) - y) ** 2)) < 1e-12),
        ("there is a classical minimum before the peak", best_before < n - 5),
        ("the far side descends by many orders of magnitude",
         test[-1] < test[peak] / 1e6),
        ("but it does not beat the classical minimum",
         test[-1] > test[best_before]),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\nall good" if all(ok for _n, ok in checks) else "\nnot yet")
    print(f"\nNow set N_TRAIN = 30 at the top and run again: the peak moves with n, "
          f"to degree ~29.\nOn a single noisy draw it can land one degree either side "
          f"of p = n (n = 30 peaks at 28 here);\naverage a few datasets and it sits "
          f"exactly on p = n. The threshold is a property of\np versus n, not of the "
          f"polynomial degree itself.")


if __name__ == "__main__":
    main()
