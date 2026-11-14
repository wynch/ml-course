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

SCRIPTS = ("gru", "ssm", "compare", "decode", "hf_compare")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_sections_1_2_same_steps():
    gru, ssm = common.read_results("gru"), common.read_results("ssm")
    steps = gru["run"]["steps"]
    for m in ssm["models"].values():
        assert m["run"]["steps"] == steps
        assert m["step_vs_parallel_max_abs_diff"] < 1e-3          # recurrence == training form
        assert abs(m["params"] / 619776 - 1) < 0.03
    assert abs(gru["params"] / 619776 - 1) < 0.001


def test_compare_equal_budget():
    r = common.read_results("compare")
    for m in r["models"].values():
        assert abs(m["run"]["train_seconds"] - r["budget_seconds"]) < 2.0   # one step of overshoot at most


def test_decode_memory():
    r = common.read_results("decode")
    for dev in r["devices"].values():
        gpt = dev["gpt"]["at"]
        assert gpt["4096"]["state_floats"] == 2 * 3 * 128 * 4096                 # KV cache: grows with t
        for k in ("gru", "selective"):
            at = dev[k]["at"]
            assert at["256"]["state_floats"] == at["4096"]["state_floats"]      # state: fixed


def test_hf_checks():
    r = common.read_results("hf_compare")
    for m in r["models"].values():
        assert m["mps_vs_cpu"]["same_top1_every_position"]
        assert m["mps_vs_cpu"]["max_abs_diff_logits"] < 1e-2
    g2, mb = r["models"]["gpt2"]["contexts"], r["models"]["mamba"]["contexts"]
    assert g2["960"]["cache_bytes_after_context"] > g2["256"]["cache_bytes_after_context"]
    assert mb["960"]["cache_bytes_after_context"] == mb["256"]["cache_bytes_after_context"]


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_window_results():
    r = common.read_results("window")
    assert r["guard"]["minutes"] < 15 and r["guard"]["peak_mps_driver_gb"] < 8
    for name, v in r["arms"].items():
        assert v["params"] == 619776                                    # a window changes no parameter
        assert sorted(x["seed"] for x in v["runs"]) == [0, 1]
        assert all(x["steps"] == r["steps"] and len(x["by_position"]) == common.BLOCK for x in v["runs"])
    c = r["copy"]
    assert c["distance"] == 48
    for name in ("w8", "w16"):                                         # reach 21 and 45 < 48: chance at best
        assert c["arms"][name]["loss"] > c["chance_loss"] - 0.05
    for name in ("full", "w64", "mix8"):
        assert c["arms"][name]["accuracy"] > 0.99
    assert r["long"]["steps"] == 4000 and set(r["long"]["arms"]) == {"full", "w8"}


def test_window_cost_results():
    r = common.read_results("window_cost")
    W = r["window"]
    d = r["decode"]
    for c in ("256", "1024", "4096"):
        n = int(c)
        assert d["full"]["at"][c]["state_floats"] == 2 * 3 * 128 * n
        assert d[f"w{W}"]["at"][c]["state_floats"] == 2 * 3 * 128 * min(n, W)
        assert d[f"mix{W}"]["at"][c]["state_floats"] == 2 * 128 * (2 * min(n, W) + n)
    t = r["train"]
    for T in ("1024", "2048", "4096"):
        assert t["window_mask"][T]["forward_alloc_bytes"] == pytest.approx(t["full"][T]["forward_alloc_bytes"], rel=0.01)
    assert t["banded"]["4096"]["ms"] * 5 < t["full"]["4096"]["ms"]
    assert t["banded"]["4096"]["forward_alloc_bytes"] * 5 < t["full"]["4096"]["forward_alloc_bytes"]


def test_window_checks_results():
    c = common.read_results("window_checks")
    assert c["guard"]["peak_mps_driver_gb"] < 8
    assert {k: [r["seed"] for r in v] for k, v in c["short"].items()} == {n: [2, 3] for n in ("full", "w8", "mix8")}
    assert {k: [r["seed"] for r in v] for k, v in c["long"]["arms"].items()} == {n: [0, 1, 2] for n in ("full", "w8")}
    assert all(r["val_step"][-1] == 4000 for v in c["long"]["arms"].values() for r in v)
    # w32 reaches 48 back only through two layers: switching off either of the first two breaks the copy
    a = c["copy_ablation"]["w32"]
    assert a["intact"]["accuracy"] > 0.95
    assert a["layer 1 attention off"]["accuracy"] < 0.1 and a["layer 2 attention off"]["accuracy"] < 0.1
