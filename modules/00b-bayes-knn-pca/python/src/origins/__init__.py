"""Module 00b — probability, neighbours, eigenvectors and clusters, from scratch.

Five small numpy implementations, no scikit-learn in the algorithms themselves
(sklearn appears only in ``tests/`` as an independent cross-check):

- :mod:`origins.bayes`  — Bayes' rule as odds updating, and Gaussian naive Bayes
- :mod:`origins.knn`    — brute-force k-nearest-neighbours + the Cover-Hart bound
- :mod:`origins.pca`    — covariance PCA by power iteration with deflation
- :mod:`origins.kmeans` — Lloyd's k-means and the moving-average (e-means) rule
- :mod:`origins.data`   — seeded synthetic data and the FashionMNIST loader
"""

__all__ = ["bayes", "knn", "pca", "kmeans", "data", "plots"]
