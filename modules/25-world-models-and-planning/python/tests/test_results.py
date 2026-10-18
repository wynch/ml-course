"""The committed results/*.json: complete, inside the laptop budget, consistent, and in the explorable.

No model is trained here; these run in a second or two.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402


@pytest.mark.parametrize("name", ["world", "models", "plan"])
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 2 and g["peak_mps_driver_gb"] == 0       # CPU only, under the unlocked-CPU cap
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_world_and_test_arenas():
    W = common.read_results("world")
    assert W["grid"] == 8 and W["obs_dim"] == 192
    assert abs(W["raw_arenas"]["wall_share_mean"] - 0.2) < 0.01
    t = W["test_arenas"]
    assert t["n"] == 300 and t["min_dist"] == 5 and t["max_dist"] == 20 and t["drawn"] >= 300


def test_models_numbers():
    R = common.read_results("models")
    assert R["data"]["train_transitions"] == 100_000 and R["data"]["held_transitions"] == 12_500
    assert 0 <= R["onehot"]["next_cell_acc_held_out"] <= 1
    for k, v in R["rollout_cell_acc"].items():
        assert len(v) == 8, k
    assert set(R["keeps"]["probe_r2"]) == {"latent", "latent_random", "raw_observation"}
    assert R["keeps"]["probe_r2"]["raw_observation"]["goal_offset"] > 0.99    # the input holds the offset linearly


def test_plan_ceiling_and_agents():
    P = common.read_results("plan")
    A = P["agents"]
    assert set(A) == {"bfs", "onehot", "latent", "latent_random", "random", "perfect"}
    assert A["bfs"]["success"] == 1.0                    # the filter keeps only goals within 20 steps
    assert A["perfect"]["success"] >= A["onehot"]["success"] - 0.02
    assert P["arenas"]["n"] == 300 and sum(b["n"] for b in A["bfs"]["by_distance"].values()) == 300
    assert P["shots"] == 64 and P["horizon"] == 8 and P["budget"] == 20


def test_artifact_present():
    assert (common.ARTIFACTS / "models.pt").stat().st_size < 1_000_000


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
