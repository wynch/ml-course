"""Solution 2 — distance-weighted k-NN.

The weighting is one `np.where`; the tally is one comprehension over the
classes. The lesson is in the printed table: at k=1 weighting cannot change
anything (there is one vote), and at every larger k it is slightly *worse* than
the plain majority — see the note the check prints.

Run:  cd python && uv run ../solutions/ex2_knn_weighted.py
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

        if self.weights == "uniform":
            w = np.ones_like(dist)
        else:
            # the epsilon stops an exact duplicate point voting infinitely
            w = 1.0 / (dist + 1e-12)

        votes = np.stack(
            [np.where(labels == c, w, 0.0).sum(axis=1) for c in self.classes_], axis=1
        )
        return self.classes_[votes.argmax(axis=1)]

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
