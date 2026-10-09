"""Exercise 1 — write the log-likelihood of Gaussian naive Bayes.

`fit` is done for you: it stores a per-class mean, a per-class variance and a
log prior. What is missing is the scoring function — the line that turns those
statistics back into "how much does class c like this point?".

Fill in the `# TODO(you):` block in `joint_log_likelihood`. The self-check at
the bottom compares your accuracy against the module's own implementation on
the same seeded data, and separately checks a single hand-computed number.

Run:  cd python && uv run ../exercises/ex1_naive_bayes.py
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
            # TODO(you): fill out[:, i].
            #
            # The 1-D Gaussian log-density of one feature is
            #     -0.5 * log(2*pi*var) - (x - mu)**2 / (2*var)
            # "Naive" means the features are assumed independent given the
            # class, so you SUM that over all D features and add the log prior.
            #
            # One vectorised line for the squared-z sum and one for the
            # normalising constant is enough — no python loop over features.
            #
            # out[:, i] = ???
            raise NotImplementedError("implement joint_log_likelihood")
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
