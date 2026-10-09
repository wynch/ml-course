"""Exercise 2 — distance-weighted k-NN, and does it actually help?

Plain k-NN gives the 1st and the k-th neighbour an equal vote, even when one is
touching the query point and the other is halfway across the plane. The usual
fix is to weight each vote by 1/distance.

Fill in the two `# TODO(you):` blocks: the weights, and the vote tally. Then
read the table the check prints and answer the question it asks — the
interesting result is not that your code runs, it is that on this data the
"obvious improvement" makes things slightly worse.

Run:  cd python && uv run ../exercises/ex2_knn_weighted.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from origins.data import two_gaussians
from origins.knn import KNN as Reference
from origins.knn import bayes_error_two_gaussians

SEP, SIGMA = 2.0, 1.0


class MyKNN:
    def __init__(self, k=5, weights="uniform"):
        self.k = int(k)
        self.weights = weights

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # squared euclidean distances, (N_query, N_train), no python loop
        d2 = ((X ** 2).sum(1)[:, None] - 2.0 * X @ self.X_.T + (self.X_ ** 2).sum(1)[None, :])
        np.maximum(d2, 0.0, out=d2)
        idx = np.argsort(d2, axis=1)[:, : self.k]          # (N, k) neighbour indices
        dist = np.sqrt(np.take_along_axis(d2, idx, axis=1))  # (N, k) distances
        labels = self.y_[idx]                                # (N, k) neighbour labels

        # TODO(you): build `w`, the (N, k) array of vote weights.
        #   'uniform'  -> every neighbour votes 1
        #   'distance' -> every neighbour votes 1 / (its distance + 1e-12)
        #                 (the epsilon stops an exact duplicate point from
        #                  producing an infinite vote)
        # w = ???
        raise NotImplementedError("build the weights")

        # TODO(you): tally the weights per class and return the winning label.
        # For each class c in self.classes_, sum w where labels == c, then take
        # the argmax across classes. `np.where(labels == c, w, 0.0).sum(axis=1)`
        # gives you one class's column.
        # votes = ???
        # return self.classes_[votes.argmax(axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


def _check():
    Xtr, ytr = two_gaussians(1000, sep=SEP, sigma=SIGMA, seed=1)
    Xte, yte = two_gaussians(20_000, sep=SEP, sigma=SIGMA, seed=2)
    R_star = bayes_error_two_gaussians(SEP, SIGMA)

    ok = True
    print(f"Bayes error for this data: {R_star:.4f}\n")
    print("   k   uniform   distance   reference(uniform)")
    for k in (1, 5, 25, 101):
        u = 1 - MyKNN(k=k, weights="uniform").fit(Xtr, ytr).score(Xte, yte)
        d = 1 - MyKNN(k=k, weights="distance").fit(Xtr, ytr).score(Xte, yte)
        r = 1 - Reference(k=k, weights="uniform").fit(Xtr, ytr).score(Xte, yte)
        print(f"  {k:3d}   {u:.4f}    {d:.4f}     {r:.4f}")
        ok = ok and abs(u - r) < 1e-12

    print("\n" + ("PASS — your uniform k-NN matches the module's exactly."
                  if ok else "FAIL — implement the two TODO blocks above."))
    print("Now read the table. At k=1 the two columns are identical — there is\n"
          "only one vote to weight. At k>1 weighting is consistently WORSE here.\n"
          "Why: this data's true posterior eta(x) is smooth, so a big flat\n"
          "neighbourhood is a good estimator of it, and 1/d weighting hands the\n"
          "decision back to the one or two closest (noisiest) labels. Weighting\n"
          "pays off when the posterior changes fast compared to the neighbour\n"
          "spacing — not here.")
    return ok


if __name__ == "__main__":
    _check()
