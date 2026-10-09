"""The dual SVM: is the projection right, are the KKT conditions met, and does
the answer match a battle-tested solver?"""

import numpy as np
import pytest

from kernelmem.lift import check_identity, phi_quadratic
from kernelmem.svm import DualSVM, circles, kernel_matrix, project_feasible, separable_blobs


def test_projection_is_feasible_and_optimal():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(4, 30)) * 2
        y = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])
        v = rng.normal(size=n) * 3.0
        a = project_feasible(v, y)
        assert (a >= -1e-12).all()
        assert abs(a @ y) < 1e-9
        # optimality: no feasible point is closer to v than the projection
        for _ in range(30):
            other = np.abs(rng.normal(size=n))
            other = project_feasible(other, y)
            assert np.linalg.norm(v - a) <= np.linalg.norm(v - other) + 1e-9


def test_projection_respects_the_box():
    y = np.array([1.0, 1.0, -1.0, -1.0])
    a = project_feasible(np.array([5.0, -2.0, 3.0, 0.4]), y, C=1.0)
    assert (a >= -1e-12).all() and (a <= 1.0 + 1e-12).all()
    assert abs(a @ y) < 1e-9


def test_kkt_and_margin_identity():
    X, y = separable_blobs()
    m = DualSVM(kernel="linear", iters=4000).fit(X, y)
    sv = m.support_mask()

    # every support vector sits exactly on the margin
    f = y * m.decision_function(X)
    assert np.allclose(f[sv], 1.0, atol=1e-4)
    # and nothing sits inside it
    assert (f >= 1.0 - 1e-4).all()
    # the equality constraint holds
    assert abs(m.alpha @ y) < 1e-9
    # ||w||^2 == sum(alpha) at the optimum, so the two margin formulas agree
    w = m.weights()
    assert m.margin() == pytest.approx(2.0 / np.linalg.norm(w), rel=1e-6)
    assert float(w @ w) == pytest.approx(float(m.alpha.sum()), rel=1e-6)
    # sparsity: this is the whole point of the dual
    assert int(sv.sum()) == 3


def test_matches_sklearn_on_the_blobs():
    sklearn_svm = pytest.importorskip("sklearn.svm")
    X, y = separable_blobs()
    ours = DualSVM(kernel="linear", iters=4000).fit(X, y)
    ref = sklearn_svm.SVC(kernel="linear", C=1e6, tol=1e-12).fit(X, y)

    assert np.allclose(ours.weights(), ref.coef_[0], atol=1e-6)
    assert ours.b == pytest.approx(float(ref.intercept_[0]), abs=1e-6)
    assert set(np.where(ours.support_mask())[0]) == set(ref.support_.tolist())
    assert ours.margin() == pytest.approx(2.0 / np.linalg.norm(ref.coef_[0]), rel=1e-6)


def test_kernel_trick_equals_the_explicit_lift():
    X, _y = circles(n=40, seed=1)
    assert check_identity(X) < 1e-10
    F = phi_quadratic(X)
    assert np.allclose(kernel_matrix(X, X, "poly", degree=2), F @ F.T, atol=1e-10)


def test_quadratic_kernel_separates_circles_a_line_cannot():
    X, y = circles()
    poly = DualSVM(kernel="poly", params={"degree": 2}, iters=6000).fit(X, y)
    assert (poly.predict(X) == y).mean() == 1.0
    # the same solution, computed the slow explicit way, must agree
    w3 = (poly.alpha * y) @ phi_quadratic(X)
    assert np.allclose(phi_quadratic(X) @ w3 + poly.b, poly.decision_function(X), atol=1e-9)

    linear = DualSVM(kernel="linear", C=1.0, iters=6000).fit(X, y)
    assert (linear.predict(X) == y).mean() < 0.7
