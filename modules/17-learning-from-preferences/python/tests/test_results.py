"""The committed results/*.json: complete, inside the laptop budget, and in the explorable.

No model is loaded here; these run in a second.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402

FIVE_MIN = ("bt_synthetic", "reward_head", "dpo_by_hand", "dpo_preview", "length_bias")


@pytest.mark.parametrize("name", FIVE_MIN + ("dpo_full",))
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    # The full DPO config is the one run allowed past 5 minutes; it must stay under 15.
    assert g["minutes"] < (15 if name == "dpo_full" else 5)
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_bt_recovers_ranking():
    r = common.read_results("bt_synthetic")
    assert r["main"]["kendall_tau"] > 0.9
    assert r["main"]["loss_at_fit"] <= r["main"]["loss_at_true"] < r["main"]["loss_at_zero"]
    taus = [s["kendall_tau_mean"] for s in r["sweep"]]
    assert taus == sorted(taus)                                   # more comparisons, better recovery


def test_reward_head():
    r = common.read_results("reward_head")
    assert r["data"]["train_pairs"] == 2000 and r["data"]["heldout_pairs"] == 500
    assert r["early_stop_vs_full_pass_max_abs_diff"] < 1e-3
    assert r["mps_vs_cpu"]["same_top1_every_position"]
    assert len(r["head"]["w"]) == r["hidden_size"] == 960
    assert len(r["heldout_scores"]) == 500


def test_dpo_by_hand_matches_trl():
    r = common.read_results("dpo_by_hand")
    assert r["abs_diff"]["loss"] < 1e-3
    assert all(abs(p["loss_policy_equals_reference"] - math.log(2)) < 1e-12 for p in r["per_pair"])
    assert r["hand"]["loss"] != pytest.approx(math.log(2), abs=1e-3)   # the random adapter moved the policy


@pytest.mark.parametrize("cfg", ("preview", "full"))
def test_dpo_runs(cfg):
    r = common.read_results(f"dpo_{cfg}")
    h = r["heldout"]
    assert h["implicit_reward_accuracy_before"]["ties"] == h["pairs"]  # policy = reference at step 0
    assert h["adapter_off_check_max_abs_diff"] < 1e-2
    assert len(h["per_pair"]) == h["pairs"]
    assert r["settings"]["beta"] == 0.1
    assert len(r["generations"]["samples"]) == r["generations"]["prompts"]
    assert r["settings"]["steps"] == (200 if cfg == "full" else r["settings"]["steps"])


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
