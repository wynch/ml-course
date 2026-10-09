"""Cross-checks for module 00b.

The algorithms in ``src/origins`` are written from scratch on numpy. These tests
hold them against three independent references: scikit-learn's implementations,
``numpy.linalg.eigh``, and closed-form probability.

Run:  cd python && uv run pytest
"""

import numpy as np
import pytest

from origins.bayes import (
    GaussianBayes,
    GaussianNaiveBayes,
    likelihood_ratio,
    odds,
    posterior,
    prob,
    sequential_posteriors,
)
from origins.data import anisotropic_blob, gaussian_pair_2d, two_gaussians
from origins.knn import (
    KNN,
    asymptotic_1nn_error,
    bayes_error_two_gaussians,
    cover_hart_bound,
    eta_two_gaussians,
)
from origins.pca import PCA, _fix_signs, power_iteration, top_eigenpairs


# ───────────────────────────── Bayes ─────────────────────────────


def test_posterior_matches_hand_arithmetic():
    # 0.004*0.96 / (0.004*0.96 + 0.996*0.08)
    assert posterior(0.004, 0.96, 0.92) == pytest.approx(0.00384 / 0.08352, rel=1e-12)


def test_sequential_updating_is_odds_multiplication():
    lr = likelihood_ratio(0.96, 0.92)
    assert lr == pytest.approx(12.0, rel=1e-12)
    chain = sequential_posteriors(0.004, 0.96, 0.92, [True] * 4)
    for k, p in enumerate(chain, start=1):
        assert p == pytest.approx(prob(odds(0.004) * lr ** k), rel=1e-12)


def test_negative_result_lowers_the_posterior():
    p = posterior(0.004, 0.96, 0.92, positive=False)
    assert p < 0.004
    assert p == pytest.approx(0.004 * 0.04 / (0.004 * 0.04 + 0.996 * 0.92), rel=1e-12)


def test_naive_bayes_matches_sklearn():
    sk = pytest.importorskip("sklearn.naive_bayes")
    X, y = gaussian_pair_2d(400, seed=3)
    Xt, yt = gaussian_pair_2d(500, seed=11)
    ours = GaussianNaiveBayes(var_smoothing=1e-9).fit(X, y)
    ref = sk.GaussianNB(var_smoothing=1e-9).fit(X, y)
    np.testing.assert_allclose(ours.theta_, ref.theta_, rtol=1e-12)
    np.testing.assert_allclose(ours.var_, ref.var_, rtol=1e-10)
    np.testing.assert_array_equal(ours.predict(Xt), ref.predict(Xt))
    np.testing.assert_allclose(ours.predict_proba(Xt), ref.predict_proba(Xt), atol=1e-10)


def test_naive_bayes_probabilities_sum_to_one():
    X, y = gaussian_pair_2d(300, seed=3)
    p = GaussianNaiveBayes().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)


def test_full_covariance_beats_naive_on_correlated_data():
    """The demo's headline claim, asserted rather than eyeballed."""
    X, y = gaussian_pair_2d(400, seed=3)
    Xt, yt = gaussian_pair_2d(2000, seed=11)
    naive = GaussianNaiveBayes().fit(X, y).score(Xt, yt)
    full = GaussianBayes().fit(X, y).score(Xt, yt)
    assert full > naive
    assert naive == pytest.approx(0.832, abs=1e-9)
    assert full == pytest.approx(0.893, abs=1e-9)


def test_full_covariance_matches_sklearn_qda():
    sk = pytest.importorskip("sklearn.discriminant_analysis")
    X, y = gaussian_pair_2d(400, seed=3)
    Xt, _ = gaussian_pair_2d(500, seed=11)
    ours = GaussianBayes(var_smoothing=0.0).fit(X, y)
    ref = sk.QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=0.0).fit(X, y)
    np.testing.assert_array_equal(ours.predict(Xt), ref.predict(Xt))


# ───────────────────────────── k-NN ─────────────────────────────


@pytest.mark.parametrize("k", [1, 5, 25])
def test_knn_matches_sklearn(k):
    sk = pytest.importorskip("sklearn.neighbors")
    X, y = two_gaussians(500, sep=2.0, seed=1)
    Xt, _ = two_gaussians(400, sep=2.0, seed=2)
    ours = KNN(k=k).fit(X, y).predict(Xt)
    ref = sk.KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(Xt)
    # ties are broken differently by the two libraries; k is odd so there are none
    np.testing.assert_array_equal(ours, ref)


def test_knn_k1_memorises_its_training_set():
    X, y = two_gaussians(300, sep=2.0, seed=1)
    assert KNN(k=1).fit(X, y).score(X, y) == 1.0


def test_bayes_error_closed_form():
    # sep = 2 sigma  ->  Phi(-1)
    assert bayes_error_two_gaussians(2.0, 1.0) == pytest.approx(0.15865525393145707, rel=1e-12)
    # only the ratio matters
    assert bayes_error_two_gaussians(4.0, 2.0) == pytest.approx(
        bayes_error_two_gaussians(2.0, 1.0), rel=1e-12
    )


def test_bayes_error_agrees_with_monte_carlo():
    X, _ = two_gaussians(400_000, sep=2.0, seed=99)
    eta = eta_two_gaussians(X, 2.0, 1.0)
    mc = float(np.minimum(eta, 1 - eta).mean())
    assert mc == pytest.approx(bayes_error_two_gaussians(2.0, 1.0), abs=2e-3)


def test_cover_hart_sandwich():
    """R* <= asymptotic 1-NN error <= 2 R* (1 - R*) <= 2 R*."""
    X, _ = two_gaussians(400_000, sep=2.0, seed=99)
    eta = eta_two_gaussians(X, 2.0, 1.0)
    R = bayes_error_two_gaussians(2.0, 1.0)
    lim = asymptotic_1nn_error(eta)
    bound = cover_hart_bound(R, 2)
    assert R <= lim <= bound <= 2 * R
    assert bound == pytest.approx(2 * R * (1 - R), rel=1e-12)


def test_measured_1nn_error_respects_the_bound():
    X, y = two_gaussians(4000, sep=2.0, seed=1000)
    Xt, yt = two_gaussians(8000, sep=2.0, seed=2)
    err = 1.0 - KNN(k=1).fit(X, y).score(Xt, yt)
    R = bayes_error_two_gaussians(2.0, 1.0)
    assert R < err < cover_hart_bound(R, 2)


# ───────────────────────────── PCA ─────────────────────────────


def test_power_iteration_finds_the_dominant_eigenpair():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(8, 8))
    A = B @ B.T
    lam, v, n = power_iteration(A, tol=1e-13)
    w, V = np.linalg.eigh(A)
    assert lam == pytest.approx(w[-1], rel=1e-10)
    assert abs(float(v @ V[:, -1])) == pytest.approx(1.0, abs=1e-8)
    assert 1 <= n < 10_000


def test_power_iteration_handles_a_negative_dominant_eigenvalue():
    A = np.diag([-5.0, 1.0, 0.5])
    lam, v, _ = power_iteration(A, tol=1e-13, seed=1)
    assert lam == pytest.approx(-5.0, rel=1e-10)
    assert abs(v[0]) == pytest.approx(1.0, abs=1e-8)


def test_deflation_recovers_the_whole_spectrum():
    rng = np.random.default_rng(3)
    B = rng.normal(size=(10, 10))
    A = B @ B.T
    vals, vecs, iters = top_eigenpairs(A, 10, tol=1e-13)
    w = np.sort(np.linalg.eigh(A)[0])[::-1]
    np.testing.assert_allclose(vals, w, rtol=1e-7, atol=1e-9)
    # the recovered vectors are orthonormal
    np.testing.assert_allclose(vecs.T @ vecs, np.eye(10), atol=1e-6)
    assert all(i >= 1 for i in iters)


def test_pca_power_matches_eigh_on_real_shaped_data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 30)) @ rng.normal(size=(30, 30))
    a = PCA(8, solver="power", tol=1e-12).fit(X)
    b = PCA(8, solver="eigh").fit(X)
    np.testing.assert_allclose(a.explained_variance_, b.explained_variance_, rtol=1e-8)
    cos = np.abs((a.components_ * b.components_).sum(axis=1))
    np.testing.assert_allclose(cos, 1.0, atol=1e-6)


def test_pca_matches_sklearn():
    sk = pytest.importorskip("sklearn.decomposition")
    rng = np.random.default_rng(6)
    X = rng.normal(size=(400, 12)) @ rng.normal(size=(12, 12))
    ours = PCA(5, solver="eigh").fit(X)
    ref = sk.PCA(n_components=5, svd_solver="full").fit(X)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-9)
    np.testing.assert_allclose(
        ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-9
    )
    np.testing.assert_allclose(
        _fix_signs(ours.components_), _fix_signs(ref.components_), atol=1e-9
    )


def test_reconstruction_error_equals_the_discarded_variance():
    """Keeping k components loses exactly the sum of the dropped eigenvalues."""
    X = anisotropic_blob(500, seed=7)
    p = PCA(2, solver="eigh").fit(X)
    Xr = p.reconstruct(X, k=1)
    mse_total = ((X - Xr) ** 2).sum(axis=1).mean()
    assert mse_total == pytest.approx(p.explained_variance_[1] * (len(X) - 1) / len(X), rel=1e-10)


def test_full_rank_reconstruction_is_exact():
    X = anisotropic_blob(200, seed=7)
    p = PCA(2, solver="power", tol=1e-13).fit(X)
    np.testing.assert_allclose(p.reconstruct(X, k=2), X, atol=1e-9)


def test_pca_finds_the_direction_the_data_was_built_with():
    X = anisotropic_blob(4000, seed=7)   # stretched (3.0, 0.8), rotated 30 degrees
    p = PCA(1, solver="power", tol=1e-13).fit(X)
    ang = np.degrees(np.arctan2(p.components_[0, 1], p.components_[0, 0])) % 180
    assert ang == pytest.approx(30.0, abs=1.5)


# ─────────────────────────── the exercises ───────────────────────────


def test_solutions_pass():
    import ex1_naive_bayes
    import ex2_knn_weighted
    import ex3_power_iteration

    assert ex1_naive_bayes._check()
    assert ex2_knn_weighted._check()
    assert ex3_power_iteration._check()
