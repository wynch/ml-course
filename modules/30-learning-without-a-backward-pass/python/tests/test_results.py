"""The committed results/*.json: complete, inside the laptop budget, consistent, and in the explorable.

Nothing is trained here; these run in about a second.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402

LIB = common.RESULTS / "library"


def jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_expectations_written_first():
    E = common.read_results("expectations")
    assert set(E["expectations"]) == {f"E{i}" for i in range(1, 13)}
    assert "before any run" in E["written"]


def test_guard_budget():
    for g in jsonl(common.RUNS / "chunks.jsonl"):
        assert g["minutes"] <= 5 and g["peak_rss_gb"] < 4 and g["peak_mps_driver_gb"] == 0
        assert g["mps_high_watermark_ratio"] == "0.5" and g["threads"] == 4
    for name in ("memory", "credit"):
        g = common.read_results(name)["guard"]
        assert g["minutes"] <= 5 and g["peak_rss_gb"] < 4
    assert common.read_results("memory")["guard"]["children_peak_rss_gb"] < 4
    for g in jsonl(LIB / "chunks.jsonl"):
        assert g["minutes"] <= 5 and g["peak_rss_gb"] < 4 and g["mps_driver_peak_gb"] < 4


def test_train_runs_complete_and_checkpointing_identical():
    T = common.read_results("train")
    assert set(T["rules"]) == {"backprop", "checkpointing", "mezo", "forward_forward", "mono_forward", "predictive_coding"}
    for r in T["rules"].values():
        assert len(r["final_test_acc"]) == 3 and len(r["mean_curve"]) == T["epochs"] == 20
    assert T["rules"]["checkpointing"]["final_test_acc"] == T["rules"]["backprop"]["final_test_acc"]
    assert T["rules"]["checkpointing"]["mean_curve"] == T["rules"]["backprop"]["mean_curve"]
    assert len(T["mezo_long"]["curve"]) == 100
    for f in common.RUNS.glob("*_s*.json"):
        assert len(json.loads(f.read_text())["curve"]) in (20, 40, 100)
    fu = T["followup"]
    assert fu["nan_from_epoch"] == [None, None, None]
    assert all(all(b > a for a, b in zip(n, n[1:])) for n in fu["weight_norms"])    # the weights' length grows
    assert T["rules"]["mezo"]["nan_from_epoch"] == [16, 16, 17]


def test_memory_numbers():
    M = common.read_results("memory")
    for d, row in M["by_depth"].items():
        r = row["rules"]
        assert r["mezo"]["extra_mb"] == 0                         # MeZO holds exactly what inference holds
        assert r["backprop"]["autograd_saved_mb"] > r["checkpointing"]["autograd_saved_mb"] or d == "2"
        assert r["predictive_coding"]["extra_mb"] > r["backprop"]["extra_mb"]
    sgd = [M["by_depth"][str(d)]["rules"]["mono_forward_sgd"]["extra_mb"] for d in M["depths"]]
    assert max(sgd) - min(sgd) < 3                                 # flat in depth


def test_credit_arrives_one_layer_per_step():
    C = common.read_results("credit")
    for net in C["nets"].values():
        L = len(net["cosine"])
        assert net["first_nonzero_step"] == list(range(L - 1, -1, -1))


def test_library_lane():
    Lb = common.read_results("library", LIB)
    assert Lb["lora"]["steps"] == Lb["lora"]["config"]["steps"]
    assert Lb["mezo"]["steps"] == Lb["mezo"]["config"]["steps"]
    steps = jsonl(LIB / "mezo_steps.jsonl")
    assert len(steps) == Lb["mezo"]["steps"] and all({"seed", "g"} <= set(s) for s in steps)
    assert Lb["n_val"] == 872


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
