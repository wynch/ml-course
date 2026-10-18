"""The reference solutions, checked against src/. CPU only, a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (laptop guard before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import grid  # noqa: E402
import models as M  # noqa: E402
import plan  # noqa: E402


# ------------------------------------------------------------- (a) the world
def test_grid_step_and_bfs_match_src():
    import solution_a_grid as sol

    rng = np.random.default_rng(5)
    for _ in range(200):
        walls, start, goal = grid.make_arena(rng)
        for a in range(4):
            assert sol.step(walls, start, a) == grid.step(walls, start, a)
        assert (sol.bfs(walls, goal) == grid.bfs(walls, goal)).all()


def test_grid_rules():
    walls = np.zeros((grid.N, grid.N), bool)
    walls[0, 1] = True
    assert grid.step(walls, (0, 0), 0) == (0, 0)              # off the grid: stay
    assert grid.step(walls, (0, 0), 3) == (0, 0)              # into a wall: stay
    assert grid.step(walls, (0, 0), 1) == (1, 0)
    d = grid.bfs(walls, (0, 0))
    assert d[0, 2] == 4 and d[0, 1] == -1                    # around the wall; the wall itself unreached
    o = grid.observe(walls, (2, 3), (5, 6))
    assert o.shape == (192,) and o.sum() == 3 and o[64 + 2 * 8 + 3] == 1 and o[128 + 5 * 8 + 6] == 1


def test_bfs_agent_follows_a_shortest_path():
    for ar in grid.test_arenas(30, seed=11):
        dist = grid.bfs(ar["walls"], ar["goal"])
        ep = plan.run_episode("bfs", ar, np.random.default_rng(0))
        assert ep["success"] and ep["steps"] == dist[ar["start"]]


# ---------------------------------------------------------- (b) shooting
def test_shooting_matches_src_perfect_model():
    import solution_b_shooting as sol

    for i, ar in enumerate(grid.test_arenas(20, seed=3)):
        obs = grid.observe(ar["walls"], ar["start"], ar["goal"])
        a_src, _, _ = plan.shoot("perfect", None, None, obs, ar["goal"], np.random.default_rng(i))
        a_sol = sol.shoot(ar["walls"], ar["start"], ar["goal"], np.random.default_rng(i))
        assert a_src == a_sol


# ------------------------------------------------------------- (c) JEPA
def test_jepa_loss_and_ema_match_src():
    import solution_c_jepa as sol

    torch.manual_seed(0)
    m = M.JEPA()
    obs = torch.rand(16, grid.OBS_DIM)
    nxt = torch.rand(16, grid.OBS_DIM)
    a = torch.randint(4, (16,))
    assert torch.allclose(sol.jepa_loss(m, obs, a, nxt), M.jepa_loss(m, obs, a, nxt))
    t1, t2 = M.Encoder(), M.Encoder()
    t2.load_state_dict(t1.state_dict())
    online = M.Encoder()
    sol.ema_update(t1, online, 0.9)
    M.ema_update(t2, online, 0.9)
    for p, q in zip(t1.parameters(), t2.parameters()):
        assert torch.allclose(p, q)


def test_jepa_loss_sends_no_gradient_to_the_target():
    torch.manual_seed(0)
    m = M.JEPA()
    obs, nxt, a = torch.rand(8, grid.OBS_DIM), torch.rand(8, grid.OBS_DIM), torch.randint(4, (8,))
    M.jepa_loss(m, obs, a, nxt).backward()
    assert all(p.grad is None for p in m.target.parameters())
    assert all(p.grad is not None for p in m.encoder.parameters())


def test_onehot_rollout_rewrites_only_the_agent_plane():
    torch.manual_seed(0)
    m = M.OneHotModel().eval()
    ar = grid.test_arenas(1, seed=3)[0]
    obs = torch.from_numpy(grid.observe(ar["walls"], ar["start"], ar["goal"]))[None]
    cells = m.rollout(obs, torch.randint(4, (1, 5)))
    assert cells.shape == (1, 5, 2) and int(cells.min()) >= 0 and int(cells.max()) < grid.N


def test_probe_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, M.LATENT))
    W = rng.normal(size=(M.LATENT, 2))
    p = M.Probe.fit(Z, Z @ W + 3.0, lam=1e-6)
    assert np.allclose(p(Z), Z @ W + 3.0, atol=1e-4)
