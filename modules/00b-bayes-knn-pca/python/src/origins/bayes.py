"""Bayes' rule, twice: as odds updating, and as a classifier.

The first half is arithmetic you can do on paper — a screening test whose
posterior is nothing like its accuracy. The second half is the same rule with a
Gaussian likelihood per feature, which is the whole of *Gaussian naive Bayes*.
"""

from __future__ import annotations

import numpy as np

# ─────────────────────── part 1: odds updating ───────────────────────


def posterior(prior: float, sens: float, spec: float, positive: bool = True) -> float:
    """P(disease | test result) from prior, sensitivity and specificity.

    ``sens`` = P(+ | disease), ``spec`` = P(- | no disease).
    """
    if positive:
        num = prior * sens
        den = num + (1.0 - prior) * (1.0 - spec)
    else:
        num = prior * (1.0 - sens)
        den = num + (1.0 - prior) * spec
    return num / den


def likelihood_ratio(sens: float, spec: float, positive: bool = True) -> float:
    """The factor a single result multiplies the *odds* by.

    LR+ = sens / (1 - spec); LR- = (1 - sens) / spec. This is the useful form:
    Bayes' rule is a multiplication in odds space, so repeated (conditionally
    independent) results just multiply.
    """
    return sens / (1.0 - spec) if positive else (1.0 - sens) / spec


def odds(p: float) -> float:
    return p / (1.0 - p)


def prob(o: float) -> float:
    return o / (1.0 + o)


def sequential_posteriors(prior: float, sens: float, spec: float, results) -> list[float]:
    """Posterior after each result in ``results`` (True = positive test).

    Yesterday's posterior is today's prior. Assumes the tests are conditionally
    independent given the true status — see the README for why that assumption
    is doing a lot of work here.
    """
    o = odds(prior)
    out = []
    for r in results:
        o *= likelihood_ratio(sens, spec, positive=bool(r))
        out.append(prob(o))
    return out


# ─────────────────── part 2: Gaussian naive Bayes ───────────────────


class GaussianNaiveBayes:
    """Gaussian naive Bayes, fitted in closed form. No optimiser, no epochs.

    "Naive" is the independence assumption: given the class, every feature is
    modelled as its own 1-D Gaussian, so the joint log-likelihood is a plain
    sum. Fitting is therefore just a per-class mean and variance.

    ``var_smoothing`` adds a fraction of the largest feature variance to every
    variance, which is what keeps constant pixels (variance 0) from producing
    infinities on image data.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = float(var_smoothing)
        self.classes_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None   # (C, D) per-class means
        self.var_: np.ndarray | None = None     # (C, D) per-class variances
        self.log_prior_: np.ndarray | None = None  # (C,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C, D = len(self.classes_), X.shape[1]
        self.theta_ = np.zeros((C, D))
        self.var_ = np.zeros((C, D))
        self.log_prior_ = np.zeros(C)
        eps = self.var_smoothing * X.var(axis=0).max()
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + eps
            self.log_prior_[i] = np.log(len(Xc) / len(X))
        return self

    def joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """(N, C) array of ``log P(class) + sum_d log N(x_d | mu, var)``.

        The sum over features *is* the naive assumption. Written out, each term
        is ``-0.5*log(2*pi*var) - (x - mu)**2 / (2*var)``.
        """
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((len(X), len(self.classes_)))
        for i in range(len(self.classes_)):
            # one squared z-score per feature, summed — that sum is the naive bit
            z = ((X - self.theta_[i]) ** 2 / self.var_[i]).sum(axis=1)
            norm = np.log(2.0 * np.pi * self.var_[i]).sum()
            out[:, i] = self.log_prior_[i] - 0.5 * (norm + z)
        return out

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        # log-sum-exp over classes: the evidence term P(x), computed stably
        m = jll.max(axis=1, keepdims=True)
        return jll - (m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())


class GaussianBayes(GaussianNaiveBayes):
    """The same classifier with the naive assumption **dropped**.

    One full covariance matrix per class instead of a diagonal of variances, so
    each class can be a tilted ellipse. Classic quadratic discriminant analysis.
    It costs ``D(D+1)/2`` parameters per class instead of ``D`` — fine in 2-D,
    hopeless on 784 pixels with a few thousand images, which is the whole reason
    "naive" survives.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianBayes":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C, D = len(self.classes_), X.shape[1]
        self.theta_ = np.zeros((C, D))
        self.cov_ = np.zeros((C, D, D))
        self.log_prior_ = np.zeros(C)
        reg = self.var_smoothing * X.var(axis=0).max() * np.eye(D)
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(axis=0)
            self.cov_[i] = np.cov(Xc, rowvar=False) + reg
            self.log_prior_[i] = np.log(len(Xc) / len(X))
        self.var_ = np.stack([np.diag(c) for c in self.cov_])
        return self

    def joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((len(X), len(self.classes_)))
        for i in range(len(self.classes_)):
            d = X - self.theta_[i]
            sign, logdet = np.linalg.slogdet(self.cov_[i])
            # Mahalanobis distance: solve instead of inverting the matrix
            m = (d * np.linalg.solve(self.cov_[i], d.T).T).sum(axis=1)
            out[:, i] = self.log_prior_[i] - 0.5 * (
                X.shape[1] * np.log(2.0 * np.pi) + logdet + m
            )
        return out
