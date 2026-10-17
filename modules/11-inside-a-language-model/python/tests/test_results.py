"""The committed results/*.json: complete, inside the laptop budget, and in the explorable.

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

SCRIPTS = ("hooks", "probe", "patching", "poles")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_probe_controls():
    r = common.read_results("probe")
    assert len(r["accuracy_trained"]) == len(r["accuracy_random_weights"]) == 33
    assert r["plateau_accuracy"] - max(r["accuracy_random_weights"]) > 0.2
    assert abs(r["layer0_trained"] - r["majority_baseline"]) < 0.05


def test_patching_shape_and_checks():
    r = common.read_results("patching")
    assert len(r["tokens_clean"]) == len(r["tokens_corrupt"])
    assert len(r["score"]) == 33 and len(r["score"][0]) == len(r["tokens_clean"])
    assert r["mps_vs_cpu"]["same_top1_every_position"]
    assert r["mps_vs_cpu"]["max_abs_diff_logits"] < 1e-2


def test_poles_both_signs():
    r = common.read_results("poles")
    assert len(r["neurons"]) == 3
    for n in r["neurons"]:
        assert n["pos"]["max"] > 0 > n["neg"]["max"]
        for pole in ("pos", "neg"):
            cs = n[pole]["contexts"]
            assert len(cs) == 10
            assert cs[0]["share_of_max"] == 1.0
            assert all(0 < c["share_of_max"] <= 1 for c in cs)


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
