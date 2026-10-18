"""The reference solutions, checked against the scripts and against hand-worked values.

(a) and (b) run on the CPU in well under a second. (c) loads MiniLM on the
CPU from the Hugging Face cache (src/embed.py fetches it) and is skipped
without it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (MPS guard before torch)
import torch  # noqa: E402

DOCS = ["a cat sat", "a dog sat on a mat", "cats and dogs"]


# ------------------------------------------------------------------ (a) bm25
def test_bm25_by_hand_value():
    """'cat' in DOCS: N 3, df 1, lengths 3, 6, 3 so avgdl 4. Worked on paper:
    idf = ln(1 + 2.5 / 1.5) = 0.980829; norm = 1.5 (0.25 + 0.75 * 3/4) = 1.21875;
    score of doc 0 = 0.980829 * 2.5 / 2.21875 = 1.105160."""
    import bm25

    s = bm25.BM25(DOCS).scores("cat")
    assert s[0] == pytest.approx(1.105160, abs=1e-6)
    assert s[1] == 0 and s[2] == 0                       # no stemming: "cats" is another term


def test_bm25_solution_matches_script_and_scan():
    import bm25
    import solution_a_bm25 as sol

    docs = DOCS + ["The Cat, the dog and the CAT.", "Café au lait", "cafe 2016 cats"]
    postings, lengths = sol.build_index(docs)
    ref = bm25.BM25(docs)
    assert postings == ref.postings and lengths == ref.doc_len
    scan = bm25.Scan(docs)
    for q in ("cat", "the cat dog", "cafe", "café 2016", "zebra", "cat cat"):
        a, b, c = sol.bm25_scores(postings, lengths, q), ref.scores(q), scan.scores(q)
        assert a == pytest.approx(b, abs=1e-12) and b == pytest.approx(c, abs=1e-12)
    assert sol.idf(10, 10) == pytest.approx(math.log(1 + 0.5 / 10.5))


def test_tokenize_matches_the_page_rule():
    import bm25

    assert bm25.tokenize("Beyoncé's 2016 tour, U.S.!") == ["beyonce", "s", "2016", "tour", "u", "s"]


# ------------------------------------------------------------- (b) squad f1
KNOWN = [  # (prediction, gold, exact match, F1), each worked by hand
    ("Denver Broncos", "Denver Broncos", 1.0, 1.0),
    ("The Denver Broncos!", "Denver Broncos", 1.0, 1.0),           # article and punctuation dropped
    ("Broncos", "Denver Broncos", 0.0, 2 / 3),                       # P 1, R 1/2
    ("Denver Broncos won", "Denver Broncos", 0.0, 0.8),              # P 2/3, R 1
    ("cat cat", "the cat", 0.0, 2 / 3),                              # shared 1: P 1/2, R 1
    ("1,000 men", "1000 men", 1.0, 1.0),                             # the comma goes
    ("an apple", "apple", 1.0, 1.0),
    ("theatre", "the atre", 0.0, 0.0),                               # "the" only as a whole word
    ("", "Paris", 0.0, 0.0),
    ("Panthers", "Denver Broncos", 0.0, 0.0),
]


@pytest.mark.parametrize("pred,gold,em,f1", KNOWN)
def test_squad_metric_known(pred, gold, em, f1):
    import solution_b_squad_f1 as sol
    import squad_metric

    for mod in (squad_metric, sol):
        assert mod.exact(pred, gold) == em
        assert mod.f1(pred, gold) == pytest.approx(f1)


def test_squad_best_over_golds():
    import squad_metric

    assert squad_metric.best("Broncos", ["Denver Broncos", "Broncos"]) == (1.0, 1.0)


def test_rag_extract():
    import rag

    assert rag.extract("\n Answer: Denver Broncos.\nMore text") == "Denver Broncos"
    assert rag.extract("") == ""


# ------------------------------------------------------------ (c) embed
def test_mean_pool_ignores_padding():
    import embed
    import solution_c_embed_search as sol

    torch.manual_seed(0)
    h = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    out = sol.mean_pool(h, mask)
    assert torch.allclose(out, embed.mean_pool(h, mask))
    assert torch.allclose(out[0], h[0, :3].mean(0))
    P = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    hits = sol.search(P[4], P, 2)
    assert hits[0][0] == 4 and hits[0][1] == pytest.approx(1.0)


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(common.EMBED_ID, "model.safetensors",
                                             revision=common.EMBED_REVISION), str)


def test_embed_by_hand_matches_cache():
    """Solution (c) on four paragraphs, CPU, against the vectors src/embed.py stored (float16)."""
    import numpy as np

    import data
    import embed
    import solution_c_embed_search as sol

    cache = embed.cache_dir() / "minilm_paragraphs.npy"
    if not (_minilm_cached() and cache.exists()):
        pytest.skip("MiniLM or its paragraph vectors not cached; run src/embed.py first")
    d = data.load_squad()
    model, tok = common.load_embedder("cpu")
    mine = sol.embed(model, tok, [p["text"] for p in d["paragraphs"][:4]]).numpy()
    ref = np.load(cache)[:4].astype(np.float32)
    assert np.abs(mine - ref).max() < 2e-3
