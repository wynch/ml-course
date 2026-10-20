"""The reference solutions and the world's rules, checked against src/. CPU only, a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (laptop guard before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lewm as L  # noqa: E402
import plan  # noqa: E402
import tworoom as W  # noqa: E402


# ------------------------------------------------------------- (a) the world
def test_world_step_matches_src():
    import solution_a_world as sol

    rng = np.random.default_rng(3)
    pos = W.sample_free(rng, 5000)
    for _ in range(20):
        a = rng.uniform(-1.5, 1.5, (5000, 2))
        assert np.array_equal(sol.step(pos, a), W.step(pos, a))
        pos = W.step(pos, a)


def test_world_rules():
    p = np.array([[90.0, 150.0], [90.0, 49.0], [130.0, 150.0], [25.0, 25.0]], np.float32)
    a = np.array([[1, 0], [1, 0], [-1, 0], [-1, -1]], np.float32)
    nxt = p.copy()
    for _ in range(6):
        nxt = W.step(nxt, a)
    assert nxt[0, 0] == W.WALL_L - 0.5                          # blocked by the wall, from the left
    assert nxt[1, 0] > W.WALL_X                                 # through the door
    assert nxt[2, 0] == W.WALL_R + 0.5                          # blocked, from the right
    assert nxt[3, 0] == W.LO and nxt[3, 1] == W.LO              # the border
    f = W.render(p)
    assert f.shape == (4, 3, 64, 64) and float(f.min()) >= 0 and float(f.max()) <= 1


def test_geodesic_goes_through_the_door():
    s, g = np.array([[60.0, 180.0]]), np.array([[160.0, 180.0]])
    assert W.geodesic(s, g)[0] > np.hypot(*(s - g)[0]) + 50
    assert abs(W.geodesic(s, s + 30)[0] - np.hypot(30, 30)) < 1e-9


def test_expert_reaches_every_test_goal():
    for split, ts in plan.tasks().items():
        for t in ts[:20]:
            assert W.run_expert(t["start"], t["goal"], plan.BUDGET)["success"], split


# ---------------------------------------------------------- (b) SIGReg
def test_sigreg_matches_src():
    import solution_b_sigreg as sol

    torch.manual_seed(0)
    A = torch.randn(16, 256)
    A = A / A.norm(dim=0)
    for scale in (0.1, 1.0, 3.0):
        z = torch.randn(200, 16) * scale
        assert torch.allclose(sol.sigreg(z, A), L.sigreg(z, A), rtol=1e-5)


def test_lewm_loss_matches_src():
    import solution_b_sigreg as sol

    torch.manual_seed(1)
    A = torch.randn(32, 64)
    A = A / A.norm(dim=0)
    z, zp = torch.randn(64, 4, 32), torch.randn(64, 3, 32)
    ref = torch.nn.functional.mse_loss(zp, z[:, 1:]) + 0.1 * L.sigreg(z.transpose(0, 1), A)
    assert torch.allclose(sol.lewm_loss(z, zp, A), ref, rtol=1e-5)


def test_sigreg_prefers_a_standard_gaussian():
    torch.manual_seed(2)
    A = torch.randn(32, 512)
    A = A / A.norm(dim=0)
    g = torch.randn(1024, 32)
    s = {k: float(L.sigreg(z, A)) for k, z in {"gauss": g, "small": 0.1 * g, "point": torch.zeros(1024, 32)}.items()}
    assert s["gauss"] < 5 and s["small"] > 50 * s["gauss"] and s["point"] > 50 * s["gauss"]


# ------------------------------------------------------------- (c) CEM
def test_cem_matches_src():
    import solution_c_cem as sol

    world = plan.TrueWorld()
    for i, t in enumerate(plan.tasks()["same"][:5]):
        world.start(t["start"], t["goal"])
        a = sol.cem(world, torch.Generator().manual_seed(i))
        b, _ = plan.cem(world, torch.Generator().manual_seed(i))
        assert torch.allclose(a, b)


def test_shapes_of_a_training_batch():
    D = W.collect(20, 30, 0)
    f, a, p = L.sample_batch(D, np.random.default_rng(0), 8)
    assert f.shape == (8, L.SEQ, 3, 64, 64) and a.shape == (8, L.SEQ - 1, L.ACT_IN) and p.shape == (8, L.SEQ, 2)
    torch.manual_seed(0)
    m = L.WorldModel(8)
    out = L.losses(m, f, a, 0.1)
    assert set(out) == {"pred", "sigreg", "total"} and torch.isfinite(out["total"])
