"""Solution 1 — the log-likelihood of Gaussian naive Bayes.

Two vectorised lines: the summed squared z-scores, and the summed log
normalising constants. Everything else was already given.

Run:  cd python && uv run ../solutions/ex1_naive_bayes.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from origins.bayes import GaussianNaiveBayes as Reference
from origins.data import gaussian_pair_2d


class MyGaussianNB:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = float(var_smoothing)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        eps = self.var_smoothing * X.var(axis=0).max()
        self.theta_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.stack([X[y == c].var(axis=0) + eps for c in self.classes_])
        self.log_prior_ = np.array([np.log((y == c).mean()) for c in self.classes_])
        return self

    def joint_log_likelihood(self, X):
        """Return an (N, C) array: log P(class c) + sum_d log N(x_d | mu_cd, var_cd).

        Shapes you have:
            X            (N, D)
            self.theta_  (C, D)   per-class feature means
            self.var_    (C, D)   per-class feature variances
            self.log_prior_ (C,)  log of each class's share of the training set
        """
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((len(X), len(self.classes_)))
        for i in range(len(self.classes_)):
            # sum of squared z-scores over the D features
            z = ((X - self.theta_[i]) ** 2 / self.var_[i]).sum(axis=1)
            # sum of the D log normalising constants (independent of x)
            norm = np.log(2.0 * np.pi * self.var_[i]).sum()
            out[:, i] = self.log_prior_[i] - 0.5 * (norm + z)
        return out

    def predict(self, X):
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


def _check():
    Xtr, ytr = gaussian_pair_2d(400, seed=3)
    Xte, yte = gaussian_pair_2d(2000, seed=11)

    mine = MyGaussianNB().fit(Xtr, ytr)
    ref = Reference().fit(Xtr, ytr)

    # (a) the scores themselves, not just the argmax, must match
    a = mine.joint_log_likelihood(Xte)
    b = ref.joint_log_likelihood(Xte)
    err = float(np.abs(a - b).max())
    print(f"max |your log-likelihood - reference| = {err:.3e}")

    # (b) one number you can check by hand: class 0's score at its own mean
    mu, var = ref.theta_[0], ref.var_[0]
    hand = ref.log_prior_[0] - 0.5 * np.log(2 * np.pi * var).sum()
    got = float(mine.joint_log_likelihood(mu[None, :])[0, 0])
    print(f"score at class-0 mean: yours {got:+.6f}  by hand {hand:+.6f}")

    acc_mine, acc_ref = mine.score(Xte, yte), ref.score(Xte, yte)
    print(f"test accuracy: yours {acc_mine:.4f}  reference {acc_ref:.4f}")

    ok = err < 1e-9 and abs(got - hand) < 1e-9 and acc_mine == acc_ref
    print("\n" + ("PASS — that is the whole of Gaussian naive Bayes."
                  if ok else "FAIL — implement the TODO block above."))
    return ok


if __name__ == "__main__":
    _check()
