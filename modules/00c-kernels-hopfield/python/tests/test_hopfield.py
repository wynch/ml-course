"""Hopfield: energy really does only go down, stored patterns really are fixed
points, and the softmax variant really is one attention step."""

import numpy as np
import pytest

from kernelmem.hopfield import (
    async_update,
    capacity_curve,
    corrupt,
    critical_alpha,
    energy,
    glyph_patterns,
    hebbian_weights,
    modern_update,
    overlap,
)


def test_stored_patterns_are_fixed_points():
    pats, names = glyph_patterns("ATZLX")
    W = hebbian_weights(pats)
    rng = np.random.default_rng(0)
    for pat, name in zip(pats, names):
        final, _ = async_update(W, pat, rng, sweeps=4)
        assert overlap(final, pat) == pytest.approx(1.0), name


def test_energy_never_increases():
    pats, _ = glyph_patterns("ATZLX")
    W = hebbian_weights(pats)
    rng = np.random.default_rng(3)
    for i in range(len(pats)):
        start = corrupt(pats[i], 0.3, rng)
        _final, trace = async_update(W, start, rng, sweeps=6, record=True)
        es = np.array([t[1] for t in trace])
        assert (np.diff(es) <= 1e-12).all()


def test_corrupted_patterns_are_repaired():
    pats, _ = glyph_patterns("ATZLX")
    W = hebbian_weights(pats)
    rng = np.random.default_rng(4)
    ok = 0
    for i in range(len(pats)):
        for _ in range(10):
            start = corrupt(pats[i], 0.25, rng)
            final, _ = async_update(W, start, rng, sweeps=8)
            ok += overlap(final, pats[i]) >= 0.95
    # 42 of 50 at this seed. Not 50: the letters are *correlated* patterns
    # (Z and T share their two top rows), and the Hebbian rule assumes they are
    # not, so a few runs land in a mixture state instead of the target.
    assert ok >= 40


def test_weights_are_symmetric_with_no_self_coupling():
    pats, _ = glyph_patterns("ATZ")
    W = hebbian_weights(pats)
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 0.0)


def test_energy_matches_the_definition():
    pats, _ = glyph_patterns("AT")
    W = hebbian_weights(pats)
    s = pats[0]
    assert energy(W, s) == pytest.approx(float(-0.5 * s @ W @ s))


def test_capacity_collapse_brackets_the_theoretical_value():
    alphas = np.round(np.arange(0.04, 0.281, 0.01), 4)
    a, _m, frac = capacity_curve(n=200, alphas=alphas, trials=12, seed=11)
    ac = critical_alpha(a, frac)
    # far below the cliff everything comes back; far above, nothing does
    assert np.interp(0.06, a, frac) > 0.9
    assert np.interp(0.27, a, frac) < 0.2
    # the finite-size threshold sits above 0.138 and well below 0.28
    assert 0.13 < ac < 0.25


def test_modern_update_is_one_attention_step():
    pats, _ = glyph_patterns("ATZLX")
    rng = np.random.default_rng(7)
    q = corrupt(pats[2], 0.35, rng)
    beta = 0.16

    out, w = modern_update(pats, q, beta=beta)
    # spelled out as attention: softmax(QK^T * beta) V with K = V = patterns
    logits = beta * (pats @ q)
    ref = np.exp(logits - logits.max())
    ref /= ref.sum()
    assert np.allclose(w, ref)
    assert np.allclose(out, pats.T @ ref)
    assert w.sum() == pytest.approx(1.0)
    # and at this temperature one step is enough
    assert overlap(np.sign(out), pats[2]) == pytest.approx(1.0)


def test_modern_update_beats_the_classical_capacity():
    n, p = 100, 400          # alpha = 4.0, far past the classical 0.138 cliff
    rng = np.random.default_rng(9)
    pats = rng.choice([-1.0, 1.0], size=(p, n))
    ok = 0
    for _ in range(20):
        mu = int(rng.integers(p))
        start = corrupt(pats[mu], 0.2, rng)
        out, _ = modern_update(pats, start, beta=16.0 / n)
        ok += overlap(np.sign(out), pats[mu]) >= 0.95
    assert ok == 20
