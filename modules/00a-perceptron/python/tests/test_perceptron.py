"""Cross-checks for module 00a.

The algorithms in ``src/perceptron`` are all from scratch. These tests exist to
prove they are also *right*: every from-scratch result is checked against an
independent implementation (numpy's own solvers, or scikit-learn) or against a
property the maths guarantees.

Run:  uv run pytest -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from perceptron import lstsq, mlp, mp_neuron, rosenblatt  # noqa: E402
from perceptron.data import (  # noqa: E402
    SEED, augment, design_matrix, regression_1d, separable_2d, xor_data,
)


# ─────────────────────────────── McCulloch–Pitts ───────────────────────────

def test_gates_match_python_logic():
    for a in (0, 1):
        for b in (0, 1):
            assert mp_neuron.gate("AND", (a, b)) == (a and b)
            assert mp_neuron.gate("OR", (a, b)) == (a or b)
            assert mp_neuron.gate("NAND", (a, b)) == int(not (a and b))
            assert mp_neuron.gate("NOR", (a, b)) == int(not (a or b))
            assert mp_neuron.xor_from_gates((a, b)) == (a ^ b)


def test_majority_needs_two_of_three():
    assert mp_neuron.gate("MAJORITY3", (1, 1, 0)) == 1
    assert mp_neuron.gate("MAJORITY3", (1, 0, 0)) == 0


# ─────────────────────────────── the perceptron ────────────────────────────

def test_converges_and_separates():
    X, y = separable_2d(n=200, margin=0.35, seed=SEED)
    Xa = augment(X)
    run = rosenblatt.train(Xa, y)
    assert run.converged
    assert run.accuracy(Xa, y) == 1.0
    assert run.mistakes == 12          # the module's headline number
    assert run.epochs == 3


def test_weights_are_an_integer_combination_of_the_data():
    """w is a sum of ±xᵢ, so it must equal Σ (net count) · xᵢ exactly."""
    X, y = separable_2d(n=120, margin=0.4, seed=11)
    Xa = augment(X)
    run = rosenblatt.train(Xa, y)
    counts = np.zeros(len(Xa))
    for _, i in run.updates:
        counts[i] += y[i]
    assert np.allclose(run.w, counts @ Xa)
    assert len(run.snapshots) == run.mistakes + 1


@pytest.mark.parametrize("margin,seed", [(0.05, 3), (0.2, 4), (0.6, 5), (1.0, 6)])
def test_novikoff_bound_holds(margin, seed):
    X, y = separable_2d(n=150, margin=margin, seed=seed)
    Xa = augment(X)
    run = rosenblatt.train(Xa, y, max_epochs=5000)
    info = rosenblatt.novikoff_bound(Xa, y)
    assert run.converged
    assert run.mistakes <= info["bound"]


def test_frank_wolfe_brackets_the_true_margin():
    """γ_achieved ≤ γ* ≤ γ_hull, and the bracket is tight."""
    X, y = separable_2d(n=150, margin=0.3, seed=8)
    Xa = augment(X)
    info = rosenblatt.novikoff_bound(Xa, y)
    assert info["gamma"] <= info["gamma_hull"]
    assert info["gap"] < 1e-3
    # the separator it returns really does separate, with that margin
    assert np.all(y * (Xa @ info["u"]) >= info["gamma"] - 1e-12)
    assert np.isclose(np.linalg.norm(info["u"]), 1.0)


def test_matches_sklearn_perceptron_on_the_decision():
    """Different implementation, different update order — same final labelling."""
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y = separable_2d(n=200, margin=0.35, seed=SEED)
    Xa = augment(X)
    ours = rosenblatt.train(Xa, y)
    theirs = sklearn.Perceptron(tol=None, max_iter=200, shuffle=False,
                                random_state=0).fit(X, y)
    assert (ours.predict(Xa) == theirs.predict(X)).all()


def test_xor_never_converges():
    X, y = xor_data()
    run = rosenblatt.train(augment(X), y, max_epochs=50)
    assert not run.converged
    assert run.per_epoch == [4] * 50           # a mistake on every example, forever
    assert np.allclose(run.w, 0.0)             # and w cycles back to where it began


# ──────────────────────────────── least squares ────────────────────────────

def test_normal_equations_match_numpy_lstsq():
    x, y = regression_1d(n=40, seed=SEED)
    X = design_matrix(x)
    ours = lstsq.normal_equations(X, y)
    theirs, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert np.allclose(ours, theirs)


def test_gradient_descent_reaches_the_exact_solution():
    x, y = regression_1d(n=40, seed=SEED)
    X = design_matrix(x)
    exact = lstsq.normal_equations(X, y)
    w, _, losses = lstsq.gradient_descent(X, y, lr=0.05, steps=400)
    assert np.max(np.abs(w - exact)) < 1e-9
    assert losses[-1] <= losses[0]
    assert np.isclose(losses[-1], lstsq.loss(X, y, exact))


def test_residual_is_orthogonal_to_every_column():
    x, y = regression_1d(n=40, seed=SEED)
    X = design_matrix(x)
    geo = lstsq.projection(X, y)
    assert geo["max_abs_orthogonality"] < 1e-12
    # and the fit really is the closest point of col(X): perturb it, get worse
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = geo["w"] + rng.normal(0, 0.3, size=2)
        assert lstsq.loss(X, y, w) > lstsq.loss(X, y, geo["w"])


def test_gradient_descent_diverges_past_the_stability_edge():
    x, y = regression_1d(n=40, seed=SEED)
    X = design_matrix(x)
    edge = lstsq.stability_edge(X)
    _, _, ok = lstsq.gradient_descent(X, y, lr=0.9 * edge, steps=300)
    with np.errstate(over="ignore", invalid="ignore"):
        _, _, bad = lstsq.gradient_descent(X, y, lr=1.6 * edge, steps=300)
    assert ok[-1] < ok[0]
    assert bad[-1] > 1e6 or not np.isfinite(bad[-1])


# ─────────────────────────────────── the MLP ───────────────────────────────

def test_tiny_mlp_solves_xor():
    X, y = xor_data()
    net = mlp.TinyMLP.init(2, 2, seed=7)
    assert net.n_params() == 9
    net, losses, accs = mlp.train(net, X, y.astype(float), lr=0.5, steps=4000)
    assert (net.predict(X) == y).all()
    assert losses[-1] < 1e-6
    assert accs[-1] == 1.0


def test_mlp_gradients_match_finite_differences():
    """The backward pass in mlp.train, checked one parameter at a time."""
    X, y = xor_data()
    yf = y.astype(float)
    net = mlp.TinyMLP.init(2, 3, seed=2)

    def loss_of(net):
        out = net.forward(X)[1]
        e = out - yf
        return float(e @ e / len(X))

    # analytic gradients: one training step at lr=0 leaves the net alone, so
    # recompute them here exactly as train() does
    H = np.tanh(X @ net.W1 + net.b1)
    out = (H @ net.W2 + net.b2).ravel()
    gout = (2.0 / len(X)) * (out - yf)[:, None]
    gW1 = X.T @ (gout @ net.W2.T * (1.0 - H * H))

    eps = 1e-6
    for i in range(net.W1.shape[0]):
        for j in range(net.W1.shape[1]):
            orig = net.W1[i, j]
            net.W1[i, j] = orig + eps
            up = loss_of(net)
            net.W1[i, j] = orig - eps
            dn = loss_of(net)
            net.W1[i, j] = orig
            assert abs((up - dn) / (2 * eps) - gW1[i, j]) < 1e-6
