"""The committed results/*.json: complete, inside the laptop budget, and in the explorable.

No model is loaded here. The rescoring test needs the SQuAD file in the
Hugging Face cache (any script fetches it) and is skipped without it.
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

SCRIPTS = ("embed", "retrieval", "rag")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_embed_checks():
    r = common.read_results("embed")
    assert r["data"]["n_paragraphs"] == 2067 and r["data"]["n_questions"] == 10570
    assert r["mps_vs_cpu"]["max_abs_diff"] < 1e-4
    for m in r["models"].values():
        assert sum(m["hist"]["gold"]) <= 10570 and 0 <= m["nn_same_article"] <= 1
        assert m["cos_question_gold"]["median"] > m["cos_question_random"]["median"]


def test_retrieval_tables():
    r = common.read_results("retrieval")
    rec = r["recall"]
    assert rec["gold"] == {"@1": 1.0, "@5": 1.0, "@20": 1.0}          # the ceiling row
    for n in ("bm25", "minilm", "smollm2"):
        assert rec[n]["@1"] <= rec[n]["@5"] <= rec[n]["@20"] <= 1.0
        assert r["curves"][n] == sorted(r["curves"][n])
        assert r["curves"][n][0] == rec[n]["@1"] and r["curves"][n][19] == rec[n]["@20"]
    assert r["bm25"]["max_abs_diff_vs_no_index"] < 1e-4                # the index changes cost, not scores
    for t in r["gold_ties"].values():                                   # README: ties never reach the top 20
        assert t["best_rank"] is None or t["best_rank"] > 20
    o = r["top1_overlap"]
    assert sum(o.values()) == 10570
    assert (o["both"] + o["bm25_only"]) / 10570 == pytest.approx(rec["bm25"]["@1"], abs=1e-4)
    e = r["explorable"]
    assert len(e["questions"]) == len(e["passages"]) == len(e["cosine_top"]) == 200


def test_rag_rows_and_cache():
    r = common.read_results("rag")
    assert r["conditions"] == ["none", "bm25", "minilm", "gold"]       # gold, the ceiling, in every table
    assert r["mps_vs_cpu"]["same_top1_every_position"]
    gens = json.loads((common.RESULTS / r["generation_file"]).read_text())
    assert set(gens["conditions"]) == set(r["conditions"])
    assert all(len(v["reply"]) == r["n"] == 200 for v in gens["conditions"].values())
    assert gens["conditions"]["gold"]["paragraph"] != gens["conditions"]["bm25"]["paragraph"]


def test_rag_rescored_from_cache():
    """Score the cached replies again, with no model: the table must come out the same."""
    from huggingface_hub import try_to_load_from_cache

    import data
    import rag

    if not isinstance(try_to_load_from_cache(data.DATASET_ID, data.VAL_FILE, repo_type="dataset",
                                             revision=data.DATASET_REVISION), str):
        pytest.skip("SQuAD not in the Hugging Face cache; run src/data.py first")
    d = data.load_squad()
    gens = json.loads(rag.GEN_FILE.read_text())
    s = rag.score(d, data.rag_sample(d), gens)
    assert s["table"] == common.read_results("rag")["table"]


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
