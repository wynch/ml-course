"""The committed results/*.json: complete, inside the laptop budget, consistent, and in the explorable.

No model is trained or loaded here; these run in a second or two.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402
import mqar  # noqa: E402


def _guards():
    yield "capacity", common.read_results("capacity")["guard"]
    yield "forms", common.read_results("forms")["guard"]
    for k in ("qwen3", "qwen3.5"):
        yield f"library {k}", json.loads((common.RESULTS / "library" / f"{k}.json").read_text())["guard"]
    for line in (common.MQAR_RUNS / "chunks.jsonl").read_text().splitlines():
        yield "mqar chunk", json.loads(line)["guard"]


@pytest.mark.parametrize("name,g", list(_guards()))
def test_guard_budget(name, g):
    assert g["minutes"] < 5, name                          # every script and every chunk under five minutes
    assert g["peak_rss_gb"] < common.BUDGET_GB, name
    assert g["peak_mps_driver_gb"] < common.BUDGET_GB, name
    assert g.get("peak_mlx_gb", 0) < common.BUDGET_GB, name
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_capacity():
    r = common.read_results("capacity")
    assert r["delta_vs_sgd_max_abs_diff"] < 1e-12
    assert r["overwrite"]["delta"]["cos_new"] > 0.99
    for lab, c in r["curves"].items():
        assert c["recall"][0] == 1.0 and len(c["recall"]) == len(r["ns"])


def test_forms_agree():
    r = common.read_results("forms")
    for a in r["agreement"].values():
        assert a["max_abs_diff_out"] < 1e-4 and a["max_abs_diff_state"] < 1e-4
    assert r["module18_vs_decay_max_abs_diff"] < 1e-10
    for dev in r["ms_forward_backward_gated_delta"].values():
        for row in dev.values():
            assert row["chunked"] < row["recurrent"]


def test_mqar_complete_and_consistent():
    r = common.read_results("mqar")
    assert r["runs"] == len(mqar.jobs())
    for j in mqar.jobs():
        run = json.loads((common.MQAR_RUNS / f"{j['name']}.json").read_text())
        assert run["steps"] == mqar.STEPS and run["layers"] == j["layers"]
        assert run["curve"][-1]["step"] == mqar.STEPS
        assert set(run["acc"]) == {str(p) for p in mqar.PAIRS}
    assert r["max_recurrent_vs_chunked_trained"] < 1e-3          # trained models: loop == chunks
    assert r["chunk_max_minutes"] < 5
    for g in r["grid"].values():
        if g["rule"] == "attention":
            assert g["state_bytes"] == 4 * 2 * 2 * r["T"] * r["d_model"]
        else:
            assert g["state_bytes"] == 4 * 2 * r["d_model"] * g["d_head"]


def test_library():
    r = common.read_results("library")
    q3, q35 = r["models"]["qwen3"], r["models"]["qwen3.5"]
    assert q35["layer_types"].count("full_attention") == 6 and q35["layer_types"].count("linear_attention") == 18
    for T in map(str, r["contexts"]):
        assert q3["contexts"][T]["cache_bytes"] == 28 * 2 * 8 * 128 * 2 * int(T)
    assert r["bytes_per_token"]["qwen3.5"] == 6 * 2 * 2 * 256 * 2
    for m in r["models"].values():
        assert m["revision"] in ("73e3e38d981303bc594367cd910ea6eb48349da8", "da28692b5f139cb0ec58a356b437486b7dac7462")


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
