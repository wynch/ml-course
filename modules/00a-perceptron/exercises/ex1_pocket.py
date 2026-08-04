"""Exercise 1 — the pocket algorithm: a perceptron that survives noisy data.

Rosenblatt's rule is only safe on *separable* data. Flip a few labels and the
guarantee evaporates: the weights keep lurching around forever and whatever
vector you happen to hold when you run out of patience may be terrible.

The pocket algorithm (Gallant, 1990) is the one-line fix: keep training exactly
as before, but every time you update, check whether the new weights beat the
best you have seen — if so, put them "in your pocket". Return the pocket.

Your job:
  1. Implement `perceptron_epochs` (the plain rule, marked TODO(you)).
  2. Implement `pocket` on top of it.
Then run this file: it flips 6% of the labels and reports both accuracies.

Run:  cd python && uv run ../exercises/ex1_pocket.py
"""

import numpy as np

W_TRUE = np.array([1.0, 1.6])
B_TRUE = -0.55


def noisy_data(n=200, margin=0.35, flip=0.06, seed=1958):
    """Separable data with a fraction of the labels deliberately corrupted."""
    rng = np.random.default_rng(seed)
    u = W_TRUE / np.linalg.norm(W_TRUE)
    b_u = B_TRUE / np.linalg.norm(W_TRUE)
    keep = []
    while len(keep) < n:
        batch = rng.uniform(-3.0, 3.0, size=(4 * n, 2))
        keep.extend(batch[np.abs(batch @ u + b_u) >= margin])
    X = np.asarray(keep[:n])
    y = np.where(X @ u + b_u > 0, 1, -1)
    flip_idx = rng.choice(n, size=int(flip * n), replace=False)
    y[flip_idx] *= -1
    Xa = np.column_stack([X, np.ones(n)])
    return Xa, y, flip_idx


def accuracy(Xa, y, w):
    return float((np.where(Xa @ w > 0, 1, -1) == y).mean())


def perceptron_epochs(Xa, y, epochs):
    """Plain Rosenblatt. Yield `w` after every update, in order.

    Yielding rather than returning is what lets the pocket wrapper below watch
    the run without re-implementing it.
    """
    w = np.zeros(Xa.shape[1])
    for _ in range(epochs):
        for i in range(len(Xa)):
            # TODO(you): if example i is misclassified (y·(w·x) <= 0), apply the
            # update w += y[i] * Xa[i] and `yield w.copy()`.
            pass
    if False:  # keeps this a generator until your `yield` above replaces it
        yield w


def pocket(Xa, y, epochs=60):
    """Run the perceptron but return the best weights it ever held.

    Returns (w_pocket, best_accuracy, n_updates).
    """
    w_best = np.zeros(Xa.shape[1])
    acc_best = accuracy(Xa, y, w_best)
    updates = 0
    for _w in perceptron_epochs(Xa, y, epochs):
        updates += 1
        # TODO(you): score `_w`; if it beats acc_best, copy it into the pocket.
    return w_best, acc_best, updates


def main():
    Xa, y, flipped = noisy_data()
    print(f"data: 200 points, {len(flipped)} labels flipped — not separable any more")

    # the last weight vector the plain rule happens to hold
    last = np.zeros(Xa.shape[1])
    n = 0
    for w in perceptron_epochs(Xa, y, 60):
        last, n = w, n + 1
    if n == 0:
        print("\nperceptron_epochs yielded nothing — implement it first (TODO).")
        return

    w_pocket, acc_pocket, updates = pocket(Xa, y)
    print(f"plain perceptron : {updates} updates in 60 epochs, "
          f"final-weight accuracy {accuracy(Xa, y, last)*100:.1f}%")
    print(f"pocket           : accuracy {acc_pocket*100:.1f}%")
    ceiling = 1.0 - len(flipped) / len(y)
    print(f"ceiling (a perfect line on the *clean* labels): {ceiling*100:.1f}%")
    if acc_pocket >= accuracy(Xa, y, last):
        print("\n✓ the pocket is at least as good as the last iterate — "
              "usually much better.")


if __name__ == "__main__":
    main()
