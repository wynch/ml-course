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

SCRIPTS = ("regions", "intrinsic_dim", "manifold", "cka")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_regions():
    r = common.read_results("regions")
    assert r["toy"]["regions_init"] > 0 and r["toy"]["regions_trained"] > 0
    ws = r["width"]["widths"]
    assert [w["width"] for w in ws] == [32, 64, 128, 256]
    assert all(len(w["trained_counts"]) == r["width"]["lines"] for w in ws)
    assert all(w["trained_layer1_mean"] <= w["trained_mean"] for w in ws)   # layer 2 only adds cuts


def test_intrinsic_dim():
    r = common.read_results("intrinsic_dim")
    for row in r["known"]:
        if row["true_dim"] <= 10:                       # the estimators' working range at n = 10,000
            assert abs(row["twonn"] - row["true_dim"]) < 2 and abs(row["mle"] - row["true_dim"]) < 2
    f = r["fashion"]
    assert f["pixels"]["n"] == 10_000
    assert r["vit"]["mps_vs_cpu"]["max_abs_diff_cls"] < 1e-3
    assert [row["layer"] for row in r["vit"]["per_layer_cls"]] == list(range(1, 13))


def test_manifold():
    r = common.read_results("manifold")
    for key in ("curved_cube_10", "fashion_pixels"):
        c = r["pca"][key]["components"]
        assert c["0.5"] <= c["0.9"] <= c["0.95"] <= c["0.99"]


def test_cka():
    r = common.read_results("cka")
    assert abs(r["checks"]["rotated_and_scaled"] - 1) < 1e-6
    assert abs(r["checks"]["permuted"] - 1) < 1e-6
    assert all(0 <= p["cka_hidden"] <= 1 for p in r["pairs"])


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
