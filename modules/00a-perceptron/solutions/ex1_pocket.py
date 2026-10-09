"""Solution to exercise 1 — the pocket algorithm.

Run:  cd python && uv run ../solutions/ex1_pocket.py
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
    """Plain Rosenblatt, yielding w after every update."""
    w = np.zeros(Xa.shape[1])
    for _ in range(epochs):
        for i in range(len(Xa)):
            if y[i] * (Xa[i] @ w) <= 0:
                w += y[i] * Xa[i]
                yield w.copy()
    return


def pocket(Xa, y, epochs=60):
    """Run the perceptron but return the best weights it ever held."""
    w_best = np.zeros(Xa.shape[1])
    acc_best = accuracy(Xa, y, w_best)
    updates = 0
    for w in perceptron_epochs(Xa, y, epochs):
        updates += 1
        acc = accuracy(Xa, y, w)
        if acc > acc_best:
            acc_best, w_best = acc, w.copy()
    return w_best, acc_best, updates


def main():
    Xa, y, flipped = noisy_data()
    print(f"data: 200 points, {len(flipped)} labels flipped — not separable any more")

    last = np.zeros(Xa.shape[1])
    n = 0
    for w in perceptron_epochs(Xa, y, 60):
        last, n = w, n + 1

    w_pocket, acc_pocket, updates = pocket(Xa, y)
    acc_last = accuracy(Xa, y, last)
    print(f"plain perceptron : {updates} updates in 60 epochs, "
          f"final-weight accuracy {acc_last*100:.1f}%")
    print(f"pocket           : accuracy {acc_pocket*100:.1f}%")
    ceiling = 1.0 - len(flipped) / len(y)
    print(f"ceiling (a perfect line on the *clean* labels): {ceiling*100:.1f}%")
    print(f"pocket weights   : {np.round(w_pocket, 4).tolist()}")
    assert acc_pocket >= acc_last
    print("\n✓ the pocket is at least as good as the last iterate — "
          "usually much better.")
    print("Note how the plain rule never settles: with contradictory labels no "
          "line exists,\nso the updates never stop and the *last* vector you "
          "hold is an accident of when you\nstopped looking.")


if __name__ == "__main__":
    main()
