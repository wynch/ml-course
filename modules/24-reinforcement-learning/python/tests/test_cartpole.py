"""The hand-written CartPole against known values and an independent derivation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import cartpole as cp  # noqa: E402


def test_constants_match_the_source():
    """Barto, Sutton and Anderson (1983) and CartPole-v1 in Gymnasium."""
    assert (cp.GRAVITY, cp.MASS_CART, cp.MASS_POLE, cp.HALF_LENGTH) == (9.8, 1.0, 0.1, 0.5)
    assert (cp.FORCE_MAG, cp.TAU, cp.X_LIMIT, cp.MAX_STEPS) == (10.0, 0.02, 2.4, 500)
    assert cp.THETA_LIMIT == pytest.approx(math.radians(12))


def test_one_step_from_rest_exact():
    """By hand from the equations: temp = 100/11, theta_dd = -600/41, x_dd = 400/41.

    So after one push right, x_dot = 0.02 * 400/41 = 8/41 and theta_dot = -12/41,
    while x and theta stay 0 (Euler moves positions with the old speeds).
    """
    s = cp.step_state(np.zeros(4), 1)
    assert s == pytest.approx([0.0, 8 / 41, 0.0, -12 / 41], abs=1e-15)
    assert cp.step_state(np.zeros(4), 0) == pytest.approx([0.0, -8 / 41, 0.0, 12 / 41], abs=1e-15)


def test_equations_match_lagrangian():
    """Solve the two Euler-Lagrange equations of a cart with a uniform rod (half-length l)
    as a 2 x 2 linear system and compare with the closed form the module uses:

        (M + m) x_dd + m l cos(th) th_dd = F + m l th_dot^2 sin(th)
        m l cos(th) x_dd + (4/3) m l^2 th_dd = m g l sin(th)
    """
    M, m, l, g = cp.MASS_CART, cp.MASS_POLE, cp.HALF_LENGTH, cp.GRAVITY
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.uniform([-2, -3, -1.5, -4], [2, 3, 1.5, 4])
        F = rng.choice([-10.0, 10.0])
        th, thd = s[2], s[3]
        A = np.array([[M + m, m * l * math.cos(th)], [m * l * math.cos(th), 4 / 3 * m * l * l]])
        b = np.array([F + m * l * thd**2 * math.sin(th), m * g * l * math.sin(th)])
        x_dd, th_dd = np.linalg.solve(A, b)
        mine = cp.derivatives(s, F)
        assert mine[0] == pytest.approx(x_dd, rel=1e-10, abs=1e-12)
        assert mine[1] == pytest.approx(th_dd, rel=1e-10, abs=1e-12)


def test_pole_falls_the_way_it_leans():
    s = np.array([0.0, 0.0, 0.05, 0.0])
    x_dd, th_dd = cp.derivatives(s, 0.0)
    assert th_dd > 0 and x_dd < 0


def test_episode_ends_at_the_limits():
    assert cp.failed(np.array([2.41, 0, 0, 0])) and cp.failed(np.array([0, 0, -0.21, 0]))
    assert not cp.failed(np.array([2.39, 0, 0.2, 0]))
    env = cp.CartPole(np.random.default_rng(0))
    env.reset()
    env.state = np.zeros(4)
    n = 0
    while True:
        _, _, term, trunc = env.step(n % 2)            # alternate pushes keep it up for a while
        n += 1
        if term or trunc:
            break
    assert n <= cp.MAX_STEPS and (trunc == (n == cp.MAX_STEPS))


def test_vector_env_matches_single():
    rng = np.random.default_rng(3)
    vec = cp.VecCartPole(5, np.random.default_rng(1))
    states = vec.state.copy()
    for _ in range(30):
        a = rng.integers(0, 2, size=5)
        nxt, _, term, trunc, final = vec.step(a)
        for i in range(5):
            assert final[i] == pytest.approx(cp.step_state(states[i], a[i]), abs=1e-15)
        states = nxt.copy()
