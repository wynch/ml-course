"""Minimum-norm polynomial regression: does pinv actually interpolate, is the solution
actually minimum-norm, and does the spike land exactly at p = n?"""

import numpy as np
import pytest

from kernelmem import polyreg as pr


def test_dataset_is_deterministic():
    a = pr.make_dataset()
    b = pr.make_dataset()
    for u, v in zip(a, b):
        assert np.array_equal(u, v)
    x, y, xt, yt = a
    assert len(x) == pr.N_TRAIN and len(xt) == pr.N_TEST
    assert np.allclose(yt, pr.truth(xt))


def test_underparameterised_fit_is_ordinary_least_squares():
    x, y, _xt, _yt = pr.make_dataset()
    d = 5
    A = pr.legendre_features(x, d)
    beta = pr.fit_min_norm(x, y, d)
    # normal equations must hold when the design matrix is tall and full rank
    assert np.allclose(A.T @ A @ beta, A.T @ y, atol=1e-9)


def test_past_the_threshold_the_fit_interpolates_exactly():
    x, y, _xt, _yt = pr.make_dataset()
    for d in (len(x) - 1, 25, 40):
        beta = pr.fit_min_norm(x, y, d)
        assert np.allclose(pr.predict(beta, x, d), y, atol=1e-6)


def test_the_interpolant_really_is_minimum_norm():
    x, y, _xt, _yt = pr.make_dataset()
    d = 30
    A = pr.legendre_features(x, d)
    beta = pr.fit_min_norm(x, y, d)
    rng = np.random.default_rng(0)
    # add anything from the null space of A: still an interpolant, always longer
    _u, _s, vt = np.linalg.svd(A)
    null = vt[len(x):]
    for _ in range(20):
        other = beta + null.T @ rng.normal(size=null.shape[0])
        assert np.allclose(A @ other, y, atol=1e-6)
        assert np.linalg.norm(other) > np.linalg.norm(beta)


def test_the_spike_lands_on_p_equals_n():
    x, y, xt, yt = pr.make_dataset()
    n = len(x)
    sw = pr.sweep(x, y, xt, yt, range(1, 31))
    peak = int(np.argmax(sw["test"]))
    assert sw["degrees"][peak] == n - 1          # p = degree + 1 = n
    assert sw["beta_norm"][peak] == max(sw["beta_norm"])
    # the classical U really is a U
    assert sw["test"][2] < sw["test"][0]
    assert sw["test"][2] < sw["test"][11]
    # and the far side really does come back down
    assert sw["test"][29] < sw["test"][peak] / 1e6


def test_legendre_columns_are_orthonormal_on_the_interval():
    grid = np.linspace(-1, 1, 20001)
    F = pr.legendre_features(grid, 6)
    gram = (F.T @ F) * (grid[1] - grid[0])
    assert np.allclose(gram, np.eye(7), atol=2e-3)


def test_averaged_sweep_peaks_in_the_same_place():
    med = pr.sweep_averaged(range(1, 31), trials=40, seed=1)
    assert med["degrees"][int(np.argmax(med["test"]))] == pr.N_TRAIN - 1
    assert min(med["test"][:10]) == pytest.approx(med["test"][2], rel=0.6)
