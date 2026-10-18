"""Section 2: find each question's paragraph, three ways, and count how often each succeeds.

The corpus is SQuAD's 2,067 validation paragraphs; the queries are all
10,570 validation questions. A question's **gold** paragraph is the one its
writer read. Recall@k: the share of questions whose gold paragraph is among
the k paragraphs a retriever ranks first. Another paragraph may hold the
answer too, so recall against the gold paragraph slightly undercounts.

- BM25 over a hand-built inverted index (src/bm25.py);
- cosine over MiniLM vectors; cosine over SmolLM2 mean-pool vectors (both
  from src/embed.py's cache, so run that first).

No model is loaded here; it runs on the CPU in under a minute.

    uv run python src/retrieval.py --figures
"""

from __future__ import annotations

import argparse
import time

import common
import numpy as np

from bm25 import BM25, Scan
from data import load_squad, rag_sample
from embed import cache_dir

KS = (1, 5, 20)
N_BRUTE = 500           # queries timed against the no-index version


def ranks_of_gold(S: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """1-based rank of the gold paragraph in each row of scores S; a paragraph that ties it ranks above it."""
    g = S[np.arange(len(S)), gold][:, None]
    return (S >= g).sum(1)


def gold_ties(S: np.ndarray, gold: np.ndarray) -> dict:
    """Questions where another paragraph scores exactly the gold one's, and the best rank among them."""
    g = S[np.arange(len(S)), gold][:, None]
    tied = (S == g).sum(1) > 1
    r = ranks_of_gold(S, gold)[tied]
    return {"questions": int(tied.sum()), "best_rank": int(r.min()) if tied.any() else None}


def recall(ranks: np.ndarray) -> dict:
    return {f"@{k}": round(float((ranks <= k).mean()), 4) for k in KS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    g = common.Guard("retrieval")
    data = load_squad()
    paras = [p["text"] for p in data["paragraphs"]]
    qs = [q["question"] for q in data["questions"]]
    gold = np.array([q["gold"] for q in data["questions"]])

    t0 = time.perf_counter()
    index = BM25(paras)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    S_bm25 = np.array([index.scores(q) for q in qs], dtype=np.float32)
    query_ms = (time.perf_counter() - t0) / len(qs) * 1000
    g.tick("bm25")
    postings_total = sum(len(p) for p in index.postings.values())
    postings_read = float(np.mean([index.postings_read(q) for q in qs]))

    # The index changes the cost, not the scores.
    rng = np.random.default_rng(0)
    probe = rng.choice(len(qs), N_BRUTE, replace=False)
    scan = Scan(paras)
    t0 = time.perf_counter()
    brute = [scan.scores(qs[i]) for i in probe]
    brute_ms = (time.perf_counter() - t0) / N_BRUTE * 1000
    brute_diff = float(max(np.abs(np.array(b) - S_bm25[i]).max() for b, i in zip(brute, probe)))
    g.tick("brute")

    d = cache_dir()
    ranks = {"bm25": ranks_of_gold(S_bm25, gold)}
    ties = {"bm25": gold_ties(S_bm25, gold)}
    S_emb = {}
    for name in ("minilm", "smollm2"):
        P = np.load(d / f"{name}_paragraphs.npy").astype(np.float32)
        Q = np.load(d / f"{name}_questions.npy").astype(np.float32)
        S_emb[name] = Q @ P.T
        ranks[name] = ranks_of_gold(S_emb[name], gold)
        ties[name] = gold_ties(S_emb[name], gold)
    g.tick("embeddings")

    curves = {n: [round(float((r <= k).mean()), 4) for k in range(1, 21)] for n, r in ranks.items()}
    b1, m1 = ranks["bm25"] == 1, ranks["minilm"] == 1
    out = {
        "data": data["info"],
        "bm25": {"k1": index.k1, "b": index.b, "vocab": len(index.postings), "postings": postings_total,
                 "avg_doc_tokens": round(index.avgdl, 1), "build_seconds": round(build_s, 2),
                 "query_ms_index": round(query_ms, 2), "query_ms_no_index": round(brute_ms, 1),
                 "postings_read_per_query": round(postings_read, 1), "max_abs_diff_vs_no_index": brute_diff,
                 "top_idf": sorted(((t, round(v, 3)) for t, v in index.idf.items()), key=lambda x: x[1])[:5]},
        "recall": {n: recall(r) for n, r in ranks.items()} | {"gold": {f"@{k}": 1.0 for k in KS}},
        "mrr": {n: round(float((1 / r).mean()), 4) for n, r in ranks.items()},
        "median_rank": {n: int(np.median(r)) for n, r in ranks.items()},
        "gold_ties": ties,
        "curves": curves,
        "top1_overlap": {"both": int((b1 & m1).sum()), "bm25_only": int((b1 & ~m1).sum()),
                         "minilm_only": int((~b1 & m1).sum()), "neither": int((~b1 & ~m1).sum())},
    }
    for n in ranks:
        print(f"{n:8s} recall {out['recall'][n]}  MRR {out['mrr'][n]}")

    # The explorable's corpus: the 200 RAG questions and their 200 gold paragraphs.
    sample = rag_sample(data)
    sub = [data["questions"][i]["gold"] for i in sample]
    P = np.load(d / "minilm_paragraphs.npy").astype(np.float32)[sub]
    Q = np.load(d / "minilm_questions.npy").astype(np.float32)[sample]
    C = Q @ P.T
    small = BM25([paras[j] for j in sub])
    out["explorable"] = {
        "questions": [data["questions"][i]["question"] for i in sample],
        "answers": [data["questions"][i]["answers"] for i in sample],
        "titles": [data["paragraphs"][j]["title"] for j in sub],
        "passages": [paras[j] for j in sub],
        "cosine_top": [[[int(j), round(float(C[i, j]), 3)] for j in np.argsort(-C[i])[:10]] for i in range(len(sample))],
        "bm25_top": [[[j, round(s, 3)] for j, s in small.top(data["questions"][i]["question"], 10)] for i in sample],
        "full_rank": {"bm25": [int(ranks["bm25"][i]) for i in sample],
                      "minilm": [int(ranks["minilm"][i]) for i in sample]},
    }
    out["guard"] = g.report()
    common.write_results("retrieval", out)
    print(f"index: {len(index.postings)} terms, {postings_total} postings; query {query_ms:.2f} ms with index, "
          f"{brute_ms:.0f} ms without; max diff {brute_diff:.2e}")
    if args.figures:
        import figures
        figures.fig_retrieval()


if __name__ == "__main__":
    main()
