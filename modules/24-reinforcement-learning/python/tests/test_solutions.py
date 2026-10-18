"""The reference solutions, checked against the scripts and against values worked out by hand.

All run on the CPU in a few seconds; no training.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (MPS guard before torch)
import torch  # noqa: E402


# ------------------------------------------------------------- (a) bandits
@pytest.mark.parametrize("name", ("ucb_choose", "thompson_choose"))
def test_bandit_rules_match_script(name):
    import bandits
    import solution_a_bandit_rules as sol

    data = np.random.default_rng(5)
    counts = data.integers(1, 50, size=(20, 5)).astype(float)
    sums = np.floor(counts * data.random((20, 5)))
    for t in (3, 6, 100, 5000):
        r1, r2 = np.random.default_rng(t), np.random.default_rng(t)
        assert np.array_equal(getattr(sol, name)(counts, sums, t, r1), getattr(bandits, name)(counts, sums, t, r2))


def test_ucb_tries_every_arm_first():
    import solution_a_bandit_rules as sol

    z = np.zeros((3, 5))
    assert [int(sol.ucb_choose(z, z, t, np.random.default_rng(0))[0]) for t in range(1, 6)] == [0, 1, 2, 3, 4]


# ------------------------------------------------------------ (b) CartPole
def test_cartpole_step_exact():
    import solution_b_cartpole_step as sol

    assert sol.step(np.zeros(4), 1) == pytest.approx([0.0, 8 / 41, 0.0, -12 / 41], abs=1e-15)


def test_cartpole_step_matches_script():
    import cartpole as cp
    import solution_b_cartpole_step as sol

    rng = np.random.default_rng(0)
    for _ in range(100):
        s, a = rng.uniform(-1, 1, size=4), int(rng.integers(2))
        assert sol.step(s, a) == pytest.approx(cp.step_state(s, a), abs=1e-14)


# --------------------------------------------------------------- (c) clip
def test_clip_matches_script():
    import ppo
    import solution_c_ppo_clip as sol

    torch.manual_seed(0)
    new, old, adv = torch.randn(256) * 0.3, torch.randn(256) * 0.3, torch.randn(256)
    a, b = sol.clipped_objective(new, old, adv), ppo.clipped_objective(new, old, adv)
    for x, y in zip(a, b):
        assert torch.allclose(x, y)


def test_clip_by_hand():
    """Ratios 0.7, 1.0, 1.3 with advantages +1, -1, +1: min terms 0.7, -1.0, 1.2; mean 0.3; two of three outside."""
    import solution_c_ppo_clip as sol

    new = torch.log(torch.tensor([0.7, 1.0, 1.3])).requires_grad_()
    obj, frac, _ = sol.clipped_objective(new, torch.zeros(3), torch.tensor([1.0, -1.0, 1.0]))
    assert obj.item() == pytest.approx(0.3, abs=1e-6)
    assert frac.item() == pytest.approx(2 / 3)
    obj.backward()
    # d obj / d logp = ratio * A / 3 where unclipped; 0 where the clip took over (ratio 1.3, A > 0)
    assert new.grad.tolist() == pytest.approx([0.7 / 3, -1 / 3, 0.0], abs=1e-6)


def test_clip_gradient_zero_beyond_band():
    import solution_c_ppo_clip as sol

    for rho, adv in ((1.5, 1.0), (0.5, -1.0)):
        x = torch.tensor([math.log(rho)], requires_grad=True)
        sol.clipped_objective(x, torch.zeros(1), torch.tensor([adv]))[0].backward()
        assert x.grad.item() == 0.0
