"""Exercise 3 — measure the guarantee yourself.

Novikoff's theorem promises at most `(R/γ)²` mistakes. Both quantities are
things you can compute from the data before training starts, which means you
can predict the worst case and then go and see how close reality comes.

`R` is easy — the longest augmented point. `γ` is not: it is the largest margin
*any* unit separator achieves, i.e. the distance from the origin to the convex
hull of the points `yᵢ·xᵢ`. Frank–Wolfe finds it by repeatedly walking toward
the hull vertex that leans furthest away.

Your job:
  1. `radius`, `train` — a warm-up.
  2. `max_margin` — the Frank–Wolfe loop, three lines of vector arithmetic.
Then run this file: it checks the bound holds on eight datasets and reports how
much of the guarantee each run actually used.

Run:  cd python && uv run ../exercises/ex3_margin_bound.py
"""

import numpy as np

W_TRUE = np.array([1.0, 1.6])
B_TRUE = -0.55


def separable(n=200, margin=0.35, seed=1958):
    rng = np.random.default_rng(seed)
    u = W_TRUE / np.linalg.norm(W_TRUE)
    b_u = B_TRUE / np.linalg.norm(W_TRUE)
    keep = []
    while len(keep) < n:
        batch = rng.uniform(-3.0, 3.0, size=(4 * n, 2))
        keep.extend(batch[np.abs(batch @ u + b_u) >= margin])
    X = np.asarray(keep[:n])
    y = np.where(X @ u + b_u > 0, 1, -1)
    return np.column_stack([X, np.ones(n)]), y


def train(Xa, y, max_epochs=4000):
    """Return (final w, mistake count, converged?)."""
    w = np.zeros(Xa.shape[1])
    mistakes = 0
    for _ in range(max_epochs):
        before = mistakes
        for i in range(len(Xa)):
            # TODO(you): the perceptron rule — update on a mistake, count it.
            pass
        if mistakes == before:
            return w, mistakes, True
    return w, mistakes, False


def radius(Xa):
    """R = max‖x‖."""
    # TODO(you): one line.
    raise NotImplementedError


def max_margin(Xa, y, iters=20000):
    """Frank–Wolfe: the distance from 0 to the convex hull of {yᵢ·xᵢ}.

    Returns (gamma_achieved, u), where `u` is the unit separator and
    `gamma_achieved = min_i yᵢ·(u·xᵢ)`.
    """
    Z = Xa * y[:, None]
    p = Z[int(np.argmin((Z * Z).sum(axis=1)))].copy()  # start at the shortest zᵢ
    for _ in range(iters):
        # TODO(you): four steps.
        #   1. j = the index minimising Z @ p
        #   2. d = Z[j] - p                              (the search direction)
        #   3. step = clip(-(p @ d) / (d @ d), 0, 1)     (exact line search)
        #   4. p = p + step * d ; break when d is tiny or step <= 0
        break
    gamma_hull = float(np.linalg.norm(p))
    u = p / gamma_hull
    return float(np.min(y * (Xa @ u))), u


def main():
    print(f"{'seed':>6} {'margin':>7} {'R':>7} {'γ':>8} {'bound':>10} "
          f"{'mistakes':>9} {'% used':>8}")
    ok = True
    for seed in (1958, 1959):
        for margin in (0.1, 0.25, 0.5, 1.0):
            Xa, y = separable(margin=margin, seed=seed)
            _, mistakes, converged = train(Xa, y)
            assert converged, "separable data should always converge"
            R = radius(Xa)
            gamma, u = max_margin(Xa, y)
            bound = (R / gamma) ** 2
            ok &= mistakes <= bound
            print(f"{seed:6d} {margin:7.2f} {R:7.3f} {gamma:8.4f} {bound:10.1f} "
                  f"{mistakes:9d} {100*mistakes/bound:7.2f}%")
    print("\n✓ every run stayed inside the bound." if ok
          else "\n✗ a run broke the bound — your γ is too large.")


if __name__ == "__main__":
    main()
