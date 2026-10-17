"""The reference solutions, checked against src/ssl_core.py. CPU only, a few seconds."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (MPS guard before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ssl_core as S  # noqa: E402


# ------------------------------------------------------------ (a) NT-Xent
def test_nt_xent_matches_core():
    import solution_a_nt_xent as sol

    g = torch.Generator().manual_seed(0)
    z1, z2 = torch.randn(64, 32, generator=g), torch.randn(64, 32, generator=g)
    assert torch.allclose(sol.nt_xent(z1, z2, 0.2), S.nt_xent(z1, z2, 0.2), atol=1e-5)


def test_nt_xent_collapsed_is_log_2b_minus_1():
    import solution_a_nt_xent as sol

    c = torch.ones(256, 16)
    assert float(sol.nt_xent(c, c, 0.2)) == pytest.approx(math.log(511), abs=1e-4)


# ------------------------------------------------------------- (b) meters
def test_meters_match_core():
    import solution_b_meters as sol

    g = torch.Generator().manual_seed(0)
    z = torch.randn(512, 32, generator=g, dtype=torch.float64) @ torch.randn(32, 32, generator=g, dtype=torch.float64)
    ref = S.collapse_meters(z)
    assert sol.std_normalised(z) == pytest.approx(ref["std_norm"], abs=1e-4)
    assert sol.effective_rank(z) == pytest.approx(ref["erank"], abs=0.01)
    assert float(sol.variance_hinge(z.float())) == pytest.approx(float(S.vicreg_variance(z.float())), abs=1e-6)


def test_meters_tell_the_two_collapses_apart():
    import solution_b_meters as sol

    g = torch.Generator().manual_seed(1)
    point = 3 + 1e-3 * torch.randn(2048, 64, generator=g, dtype=torch.float64)
    line = 3 * torch.randn(2048, 1, generator=g, dtype=torch.float64) * torch.ones(1, 64, dtype=torch.float64)
    assert sol.std_normalised(point) < 0.01 and sol.effective_rank(point) > 50      # shrunk, but every direction
    assert sol.std_normalised(line) > 0.9 and sol.effective_rank(line) < 1.01       # spread, but one direction
    assert float(sol.variance_hinge(line)) == 0.0                                   # the hinge is satisfied on a line


# ------------------------------------------------------------- (c) SIGReg
def test_sigreg_matches_core():
    import solution_c_sigreg as sol

    z = torch.randn(256, 32, generator=torch.Generator().manual_seed(2))
    A = torch.randn(32, 64, generator=torch.Generator().manual_seed(3))
    ref = S.sigreg(z, K=64, gen=torch.Generator().manual_seed(3))
    assert float(sol.sigreg(z, A / A.norm(dim=0, keepdim=True))) == pytest.approx(float(ref), rel=1e-4)


def test_sigreg_prefers_the_gaussian():
    import solution_c_sigreg as sol

    torch.manual_seed(0)
    A = torch.randn(64, 256)
    A = A / A.norm(dim=0, keepdim=True)
    g = torch.randn(1024, 64)
    base = float(sol.sigreg(g, A))
    for bad in (0.1 * g, 3 * g, g + 1, torch.zeros(1024, 64)):
        assert float(sol.sigreg(bad, A)) > 50 * base


# ------------------------------------------------------- shared pieces
def test_encoder_size_and_shapes():
    enc = S.Encoder()
    assert 250_000 <= S.n_params(enc) <= 350_000
    x = torch.randn(2, 3, 32, 32)
    assert enc.forward_map(x).shape == (2, S.D, 8, 8) and enc(x).shape == (2, S.D)


def test_masks():
    m = S.block_mask(512, np.random.default_rng(0), "cpu")
    share = m.float().mean().item()
    assert m.shape == (512, 8, 8) and 0.3 < share < 0.7
    r = S.random_patch_mask(16, 0.6, "cpu")
    assert (r.view(16, -1).sum(1) == round(0.6 * 64)).all()
    x = torch.ones(16, 3, 32, 32)
    assert S.apply_mask(x, r).eq(0).float().mean().item() == pytest.approx(round(0.6 * 64) / 64)
