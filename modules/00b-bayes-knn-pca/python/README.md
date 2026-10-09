# mlcourse-00b-bayes-knn-pca (Python lane)

The numpy implementations for Module 00b. See the module
[README](../README.md) for the full walkthrough and every measured number.

Layout:

- `src/origins/bayes.py` — odds updating, `GaussianNaiveBayes`, `GaussianBayes`
- `src/origins/knn.py` — `KNN`, the closed-form Bayes error, the Cover-Hart bound
- `src/origins/pca.py` — `power_iteration`, `top_eigenpairs` (deflation), `PCA`
- `src/origins/data.py` — seeded generators and the FashionMNIST loader
- `src/origins/plots.py` — shared matplotlib defaults
- `scripts/` — the four labs, run with `uv run scripts/<name>.py`
- `tests/` — cross-checks against scikit-learn, `numpy.linalg.eigh` and closed forms

Quick start:

```bash
uv run scripts/bayes_screening.py   # instant
uv run scripts/naive_bayes.py       # ~30s, uses the local FashionMNIST cache
uv run scripts/knn_lab.py           # ~70s
uv run scripts/pca_lab.py           # ~45s
uv run pytest                       # ~13s
```

`scripts/knn_lab.py` and `scripts/pca_lab.py` also write `knn_results.json` and
`pca_results.json` here; the explorable inlines the scree data from the latter.

FashionMNIST is read from `~/.cache/huggingface`. If it is not cached, both
scripts detect that (`origins.data.fashion_available`) and fall back to seeded
synthetic data, printing which source they used.
