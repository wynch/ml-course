"""The reference solutions, checked against the scripts and against transformers.

All run on the CPU in a few seconds; no training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402  (MPS guard before torch)
import torch  # noqa: E402


# ---------------------------------------------------------- (a) optimisers
def test_optimisers_match_script():
    import optim_toy as ot
    import solution_a_optimisers as sol

    rng = np.random.default_rng(0)
    for mine, ref in ((sol.momentum_step, ot.momentum_step), (sol.adam_step, ot.adam_step)):
        s1, s2 = {}, {}
        p1 = p2 = rng.normal(size=2)
        for _ in range(5):
            g = rng.normal(size=2)
            p1, p2 = mine(p1, g, s1, 0.01), ref(p2, g, s2, 0.01)
        assert np.allclose(p1, p2)


def test_optimisers_steps_on_results():
    import solution_a_optimisers as sol

    r = common.read_results("optim_toy")["losses"]["rosenbrock"]["optimisers"]
    for name, fn in (("momentum", sol.momentum_step), ("adam", sol.adam_step)):
        assert sol.steps_to_target(fn, r[name]["best_lr"]) == r[name]["best_steps"]


def test_adam_matches_torch():
    import solution_a_optimisers as sol

    p0 = np.array([0.3, -0.7])
    t = torch.tensor(p0, requires_grad=True)
    opt = torch.optim.Adam([t], lr=0.05)
    p, state = p0.copy(), {}
    rng = np.random.default_rng(1)
    for _ in range(10):
        g = rng.normal(size=2)
        t.grad = torch.tensor(g)
        opt.step()
        p = sol.adam_step(p, g, state, 0.05)
    assert np.allclose(p, t.detach().numpy(), atol=1e-10)


# ------------------------------------------------------------ (b) schedule
def test_schedule_matches_transformers():
    from transformers import get_cosine_schedule_with_warmup

    import optim_mlp
    import solution_b_schedule as sol

    total, warmup = 1570, 157
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total)
    for step in range(total):
        hf = opt.param_groups[0]["lr"]
        assert sol.warmup_cosine(step, warmup, total) == pytest.approx(hf, abs=1e-12)
        assert optim_mlp.warmup_cosine(step, warmup, total) == pytest.approx(hf, abs=1e-12)
        opt.step()
        sched.step()


# ----------------------------------------------------------- (c) sharpness
def test_power_iteration_on_quadratic():
    import solution_c_sharpness as sol

    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(20, 20))
    A = Q @ torch.diag(torch.linspace(0.1, 7.0, 20)) @ Q.T
    x = torch.randn(20, requires_grad=True)
    assert sol.top_eigenvalue(lambda: 0.5 * x @ A @ x, [x]) == pytest.approx(7.0, rel=1e-3)


def test_power_iteration_matches_script():
    import landscape
    import solution_c_sharpness as sol

    torch.manual_seed(0)
    model = common.MLP((20, 16, 5))
    x, y = torch.randn(64, 20), torch.randint(0, 5, (64,))
    mine = sol.top_eigenvalue(lambda: torch.nn.functional.cross_entropy(model(x), y), list(model.parameters()))
    ref = landscape.top_eigenvalue(model, x, y)[-1]
    assert mine == pytest.approx(ref, rel=1e-4)
