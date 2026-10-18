"""The committed results/*.json: complete, inside the laptop budget, and in the explorable.

No model is loaded here, except the small line net for the browser check.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402

SCRIPTS = ("icl", "gd_demo", "fewshot", "line")
PAGE = ROOT.parents[1] / "explorables" / "20-in-context-learning.html"


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_icl_known_answer_and_control():
    r = common.read_results("icl")
    b = r["bars"]
    assert b["known_answer"]["met"] and b["known_answer"]["value"] < 1e-6
    assert b["control"]["met"]
    m = r["mse_over_d"]
    assert all(len(v) == 21 for k, v in m.items())
    assert m["net"][10] < m["net"][5] < m["net"][1]                    # error falls with pairs
    assert m["net"][10] < m["one_gd_step"][10] < m["knn_3"][10]
    assert r["mps_vs_cpu"]["max_abs_diff"] < 1e-3


def test_bars_follow_from_numbers():
    r = common.read_results("icl")
    b, m = r["bars"], r["mse_over_d"]
    assert b["n10"]["met"] == (m["net"][10] <= 0.05)
    assert b["n20"]["met"] == (m["net"][20] <= 0.02)
    assert b["n5"]["met"] == (m["net"][5] <= 1.5 * m["ridge_0.1"][5])


def test_fewshot_conditions():
    r = common.read_results("fewshot")
    c = r["conditions"]
    assert list(c) == ["k=0", "k=1", "k=4", "k=8", "k=8, random labels"]
    for v in c.values():
        assert v["n"] == 872 and v["ci95"][0] <= v["accuracy"] <= v["ci95"][1]
    assert r["mps_vs_cpu"]["same_answer"]


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node to run the page's JavaScript")
def test_explorable_net_matches_pytorch():
    """The page's own forward pass, run in Node on the demo prompt, against PyTorch's."""
    page = PAGE.read_text()
    data = re.search(r"/\* BEGIN DATA.*?\*/(.*?)/\* END DATA \*/", page, re.S).group(1)
    netjs = re.search(r"/\* BEGIN NET.*?\*/(.*?)/\* END NET \*/", page, re.S).group(1)
    demo = common.read_results("line")["demo"]
    js = (data + netjs + "\nconst f=makeNet(DATA.line.weights, DATA.line.shape);"
          f"\nconst u={json.dumps(demo['u'])}, y={json.dumps(demo['y'])}, g={json.dumps(demo['grid'])};"
          "\nconsole.log(JSON.stringify(f(u.map((v,i)=>[v,y[i]]), g)));")
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    got = json.loads(out.stdout)
    assert max(abs(a - b) for a, b in zip(got, demo["net"])) < 1e-4
