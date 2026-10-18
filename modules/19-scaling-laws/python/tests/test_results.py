"""The committed results/*.json: complete, inside the laptop budget, and in the explorable.

No model is trained here; these run in a second.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402

SCRIPTS = [f"train_d{d}" for d in common.SIZES] + ["fit", "frontier"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_runs_complete():
    for d, r in common.read_runs().items():
        seeds = sorted(x["seed"] for x in r["runs"])
        assert seeds == ([0, 1] if d in common.TWO_SEEDS else [0])
        for x in r["runs"]:
            assert [c["step"] for c in x["curve"]] == list(common.EVAL_STEPS)
        if r["device"] == "mps":
            m = r["mps_vs_cpu"]
            assert m["max_abs_diff_logits"] < 1e-2 and m["same_argmax_share"] > 0.99


def test_loss_falls_with_size_beyond_band():
    f = common.read_results("fit")
    assert all(s["drop"] > f["band"] for s in f["size_steps"])
    assert f["synthetic_check"]["alpha_seed0"] == pytest.approx(0.1, abs=0.02)


def test_frontier_rows():
    fr = common.read_results("frontier")
    assert len(fr["rows"]) == 40
    for row in fr["rows"]:
        assert row["loss"] == min(row["candidates"].values())


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
