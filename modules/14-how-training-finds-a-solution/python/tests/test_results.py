"""The committed results/*.json: complete, inside the laptop budget, true to
the grokking rule, and in the explorable. No training here; these run in a second.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402

SCRIPTS = ("optim_toy", "optim_mlp", "hf_way", "landscape", "grok_wd1", "grok_wd0")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_toy_best_is_the_minimum():
    r = common.read_results("optim_toy")
    for L in r["losses"].values():
        for o in L["optimisers"].values():
            ok = [s["steps"] for s in o["sweep"] if s["steps"] is not None]
            assert o["best_steps"] == min(ok) and o["lrs_that_reach"] == len(ok)


def test_mlp_grid_complete():
    r = common.read_results("optim_mlp")
    seed0 = [x for x in r["runs"] if x["seed"] == 0]
    assert len(seed0) == 3 * 2 * len(r["lrs"]) == 24
    for key, b in r["best"].items():
        o, s = key.split("/")
        cell = [x for x in seed0 if x["optimiser"] == o and x["schedule"] == s and not x["diverged"]]
        assert b["test_acc_seeds"][0] == max(x["test_acc"] for x in cell)


def test_hf_schedule_identical():
    r = common.read_results("hf_way")
    assert r["schedule_check"]["max_abs_diff"] < 1e-12
    assert {x["optimiser"] for x in r["runs"]} == {"sgd", "momentum", "adam"}


def test_landscape_same_target():
    r = common.read_results("landscape")
    for n in r["nets"].values():
        assert n["reached_target"] and n["train_loss"] <= r["target_train_loss"]
        assert n["top_eig_rel_change_last10"] < 1e-2                 # power iteration settled
        assert len(n["line_train"]) == len(r["line"])


@pytest.mark.parametrize("name", ("grok_wd1", "grok_wd0"))
def test_grok_rule(name):
    """The reported steps follow the rule fixed before the run."""
    g = common.read_results(name)
    assert g["rule"] == {"grokked_test_acc": 0.95, "cap_steps": 20000, "early_stop": False,
                         "eval": "every 10 steps to 500, then every 100"}
    cv = g["curve"]
    assert cv[-1]["step"] == 20000
    first = lambda key, th: next((c["step"] for c in cv if c[key] >= th), None)  # noqa: E731
    assert g["train_hits_100"] == first("train_acc", 1.0)
    assert g["test_crosses_95"] == first("test_acc", 0.95)
    assert g["grokked"] == (g["test_crosses_95"] is not None)
    assert g["n_train"] == 2823 and g["n_test"] == 6586
    if g["mps_vs_cpu"]:
        assert g["mps_vs_cpu"]["same_argmax_share"] == 1.0


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
