"""The committed results/*.json: complete, inside the laptop budget, fair, and in the explorable.

No model is loaded here; these run in a second.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402

OBJECTIVES = ("random", "supervised", "simclr", "mae", "jepa")
VARIANTS = ("no-stop-grad", "stop-grad", "ema", "vicreg-var", "sigreg")
SCRIPTS = [f"pretrain_{o}" for o in OBJECTIVES] + [f"collapse_{v}" for v in VARIANTS] + ["dinov2"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_objectives_share_encoder_and_budget():
    P = {o: common.read_results(f"pretrain_{o}") for o in OBJECTIVES}
    assert len({p["encoder_params"] for p in P.values()}) == 1
    assert 250_000 <= P["random"]["encoder_params"] <= 350_000
    trained = [P[o] for o in OBJECTIVES if o != "random"]
    assert len({p["budget_seconds"] for p in trained}) == 1
    for p in trained:
        assert p["budget_seconds"] <= p["train_seconds"] < p["budget_seconds"] + 2
        assert p["steps"] > 0
    assert P["random"]["budget_seconds"] == 0 and P["random"]["steps"] == 0      # the floor trains for zero


def test_floor_and_ceiling():
    P = {o: common.read_results(f"pretrain_{o}")["probes"] for o in OBJECTIVES}
    assert P["supervised"]["class_acc"] == max(p["class_acc"] for p in P.values())
    for p in P.values():
        assert 0.1 < p["class_acc"] < 1 and 0.25 <= p["rotation_acc"] < 1


def test_collapse_runs():
    C = {v: common.read_results(f"collapse_{v}") for v in VARIANTS}
    assert len({c["budget_seconds"] for c in C.values()}) == 1
    assert len({c["encoder_params"] for c in C.values()}) == 1
    assert C["no-stop-grad"]["final"]["std_norm"] < 0.05                         # collapsed: every latent points one way
    for c in C.values():
        assert c["log"][0]["step"] == 0 and len(c["log"]) > 10


def test_dinov2_mps_matches_cpu():
    d = common.read_results("dinov2")
    assert d["mps_vs_cpu"]["min_cosine"] > 0.999
    assert d["train_images"] == 10000 and d["test_images"] == 10000


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
