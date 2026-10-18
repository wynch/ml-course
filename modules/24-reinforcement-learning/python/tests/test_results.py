"""The committed results/*.json: complete, inside the laptop budget, consistent
with known answers, and in the explorable. No training here; these run in a second.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402

SCRIPTS = ("bandits", "qlearning", "reinforce", "ppo", "ppo_fast", "ppo_fast_noclip")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_bandits_complete_and_known_answers():
    r = common.read_results("bandits")
    assert r["pulls"] == 10_000 and r["seeds"] == 20
    for v in r["rules"].values():
        a = np.array(v["regret"])
        assert a.shape == (20, len(r["grid"])) and r["grid"][-1] == 10_000
        assert np.all(np.diff(a, axis=1) >= -1e-9)                      # pseudo-regret never falls
        assert v["final_median"] == pytest.approx(np.median(v["final_regret"]), abs=0.06)
    # random pulling loses (best - mean of the arms) per pull on average
    rnd = r["rules"]["random"]["final_median"] / r["pulls"]
    assert rnd == pytest.approx(r["random_expected_slope"], rel=0.02)
    # Lai-Robbins constant: sum of gap / KL over the losing arms
    best = max(r["means"])
    kl = lambda p, q: p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))  # noqa: E731
    assert r["lai_robbins_constant"] == pytest.approx(sum((best - p) / kl(p, best) for p in r["means"] if p < best), abs=1e-3)


def test_qlearning_known_answers():
    r = common.read_results("qlearning")
    assert r["value_iteration_matches_bfs"]
    assert r["bfs_shortest"] == 36
    assert len(r["runs"]) == 20 and all(len(x["steps"]) == r["episodes"] for x in r["runs"])
    for x in r["runs"]:
        if x["greedy_path_len"] == r["bfs_shortest"]:
            assert x["max_abs_value_error_on_greedy_path"] < 0.1        # max Q within 0.1 of minus the distance
        assert min(x["steps"]) >= r["bfs_shortest"]                     # no episode beats the shortest path


@pytest.mark.parametrize("name", ("reinforce", "ppo", "ppo_fast", "ppo_fast_noclip"))
def test_policy_gradient_runs(name):
    r = common.read_results(name)
    assert [x["seed"] for x in r["runs"]] == [0, 1, 2]
    for x in r["runs"]:
        assert x["ends"][-1] <= r["budget_steps"] and sum(x["returns"]) <= r["budget_steps"]
        assert all(1 <= v <= 500 for v in x["returns"])
        assert len(x["curve"]) == len(r["grid"]) and len(x["eval_greedy"]) == r["eval_episodes"]
        assert x["eval_greedy_mean"] == pytest.approx(np.mean(x["eval_greedy"]))
    replay = r["runs"][0]["replay"]
    assert len(replay) == r["runs"][0]["replay_len"] + 1


def test_replay_obeys_the_dynamics():
    """Each recorded state follows from the previous one by one step with some action."""
    import cartpole as cp

    r = common.read_results("ppo")
    for states in (r["runs"][0]["replay"], r["untrained_replay"]):
        s = np.array(states)
        for a, b in zip(s[:-1], s[1:]):
            assert min(np.abs(cp.step_state(a, act) - b).max() for act in (0, 1)) < 1e-3   # states stored to 4 decimals


@pytest.mark.parametrize("name,lr,clip", (("ppo", 2.5e-4, True), ("ppo_fast", 2.5e-3, True), ("ppo_fast_noclip", 2.5e-3, False)))
def test_ppo_log(name, lr, clip):
    r = common.read_results(name)
    assert r["lr"] == lr and r["use_clip"] == clip and r["budget_steps"] == common.read_results("reinforce")["budget_steps"]
    for x in r["runs"]:
        assert len(x["log"]) == r["updates"]
        assert all(0 <= u["clip_frac"] <= 1 for u in x["log"])
        assert sum(x["log"][-1]["ratio_hist_last_epoch"]) == r["n_envs"] * r["n_steps"]


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
