"""The committed results/*.json: complete, inside the laptop budget, and in the explorable.

No model is loaded here; these run in a second.
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

SCRIPTS = ("matched", "followup", "export", "kv_hand", "kv_mlx")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 4 and g["peak_mps_driver_gb"] < 4
    assert g.get("mlx_peak_gb", 0) < 4
    assert g["mps_high_watermark_ratio"] == "0.5"


@pytest.mark.parametrize("name", ("matched", "followup"))
def test_sweep_chunks(name):
    """Every chunk of the sweep stayed under 5 minutes and 4 GB."""
    rows = [json.loads(line) for line in (common.RESULTS / f"{name}_runs.jsonl").read_text().splitlines()]
    chunks = [r["chunk"] for r in rows if "chunk" in r]
    assert chunks and all(c["minutes"] < 5 for c in chunks)
    assert all(c["peak_rss_gb"] < 4 and c["peak_mps_driver_gb"] < 4 for c in chunks)
    assert common.read_results(name)["chunks"]["count"] <= len(chunks)


def test_followup_cells():
    f = common.read_results("followup")
    m = common.read_results("matched")
    assert len(f["cells"]) == len(m["cells"])
    for c, s1 in zip(f["cells"], m["cells"]):
        assert len(c["best"]["report_by_seed"]) == 3
        rates = {x["lr"] for x in c["shallow"]["rates"]}
        assert set(m["setup"]["lr_grid"]) <= rates
        assert (c["winner"] == "section 1") == (s1["report_mean"] == c["best"]["report_mean"])


def test_matched_protocol():
    import matched

    m = common.read_results("matched")
    s = m["setup"]
    assert len(m["cells"]) == len(s["budgets_mb"]) * len(s["arms"])
    for c in m["cells"]:
        assert abs(c["bytes"] / (c["budget_mb"] * 1e6) - 1) <= s["tolerance"]
        assert c["bytes"] == matched.arm_bytes(c["arm"], c["best"]["d"], c["best"]["n_layer"], s["vocab"])
        assert len(c["report_by_seed"]) == 3
        shapes = {(t["d"], t["n_layer"]) for t in c["tried"]}
        assert shapes == {tuple(x) for x in c["shapes"]}
        mid = tuple(c["shapes"][1])
        rates = {t["lr"] for t in c["tried"] if (t["d"], t["n_layer"]) == mid}
        assert set(s["lr_grid"]) <= rates
        best = min(c["tried"], key=lambda t: t["tune"])          # the choice used the tune half only
        assert (best["d"], best["n_layer"], best["lr"]) == (c["best"]["d"], c["best"]["n_layer"], c["best"]["lr"])


def test_export_rows():
    e = common.read_results("export")
    assert [r["lr"] for r in e["rows"]] == sorted(r["lr"] for r in e["rows"])
    for r in e["rows"]:
        assert 0 <= r["changed_then_reverted_by_bf16"] <= r["codes_changed"] <= e["base"]["codes"]
    m = common.read_results("matched")
    cell = next(c for c in m["cells"] if c["arm"] == "ternary" and c["budget_mb"] == 0.5)
    assert (e["base"]["d"], e["base"]["n_layer"], e["base"]["lr"]) == (cell["best"]["d"], cell["best"]["n_layer"], cell["best"]["lr"])


def test_kv_rows():
    k = common.read_results("kv_hand")
    assert k["rows"][0]["bits"] == 16 and len(k["rows"]) == 1 + 4 * 2 * 2
    assert k["ppl_tokens"] == common.PPL_WINDOWS * (common.PPL_LEN - 1)
    lib = common.read_results("kv_mlx")
    assert lib["rows"][0]["ppl"] == k["rows"][0]["ppl"]                  # a plain KVCache changes nothing
    for row in lib["cache_bytes"]:
        assert row["bf16"] == row["tokens"] * lib["bf16_bytes_per_token"]
        assert abs(row["4-bit"] / row["bf16"] - 4.5 / 16) < 1e-9          # 4 bits + a bf16 scale and bias per 64


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
