"""Module 00b — probability, neighbours and eigenvectors, from scratch.

Four small numpy implementations, no scikit-learn in the algorithms themselves
(sklearn appears only in ``tests/`` as an independent cross-check):

- :mod:`origins.bayes`  — Bayes' rule as odds updating, and Gaussian naive Bayes
- :mod:`origins.knn`    — brute-force k-nearest-neighbours + the Cover-Hart bound
- :mod:`origins.pca`    — covariance PCA by power iteration with deflation
- :mod:`origins.data`   — seeded synthetic data and the FashionMNIST loader
"""

__all__ = ["bayes", "knn", "pca", "data", "plots"]
