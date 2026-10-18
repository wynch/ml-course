"""SQuAD v1.1 validation: paragraphs, questions, and the samples the module uses.

The validation split is one 1.8 MB Parquet file of 10,570 questions. Each row
holds a question, the paragraph (``context``) it was written on, the Wikipedia
article title, and the answer spans three crowd workers gave. Several
questions share a paragraph, so we deduplicate the paragraphs into the corpus
the retrievers search; each question keeps the index of its own paragraph,
its **gold** paragraph.

SQuAD is licensed CC BY-SA 4.0 (Rajpurkar et al., 2016); its paragraphs come
from Wikipedia.

    uv run python src/data.py      # fetch, print the counts
"""

from __future__ import annotations

import numpy as np

DATASET_ID = "rajpurkar/squad"
DATASET_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
VAL_FILE = "plain_text/validation-00000-of-00001.parquet"

N_RAG = 200         # section 3 and the explorable: questions, each on a different paragraph
SEED = 0


def load_squad() -> dict:
    """Paragraphs (deduplicated, in file order) and questions, each with its gold paragraph."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET_ID, VAL_FILE, repo_type="dataset", revision=DATASET_REVISION)
    rows = pq.read_table(path).to_pylist()
    para_index: dict[str, int] = {}
    paragraphs: list[dict] = []
    questions: list[dict] = []
    for r in rows:
        ctx = r["context"]
        if ctx not in para_index:
            para_index[ctx] = len(paragraphs)
            paragraphs.append({"id": len(paragraphs), "title": r["title"], "text": ctx})
        questions.append({"id": r["id"], "question": r["question"].strip(), "gold": para_index[ctx],
                          "answers": list(r["answers"]["text"])})
    return {"paragraphs": paragraphs, "questions": questions,
            "info": {"dataset": DATASET_ID, "revision": DATASET_REVISION, "file": VAL_FILE,
                     "licence": "CC BY-SA 4.0", "n_paragraphs": len(paragraphs),
                     "n_questions": len(questions),
                     "n_articles": len({p["title"] for p in paragraphs})}}


def rag_sample(data: dict) -> list[int]:
    """N_RAG question indices on N_RAG different paragraphs: shuffle the paragraphs, take one question each."""
    rng = np.random.default_rng(SEED)
    by_para: dict[int, list[int]] = {}
    for i, q in enumerate(data["questions"]):
        by_para.setdefault(q["gold"], []).append(i)
    paras = rng.permutation(len(data["paragraphs"]))[:N_RAG]
    return [int(by_para[int(p)][rng.integers(len(by_para[int(p)]))]) for p in paras]


if __name__ == "__main__":
    d = load_squad()
    print(d["info"])
    lens = [len(p["text"].split()) for p in d["paragraphs"]]
    print(f"paragraph words: median {int(np.median(lens))}, max {max(lens)}")
    s = rag_sample(d)
    print(f"RAG sample: {len(s)} questions on {len({d['questions'][i]['gold'] for i in s})} paragraphs")
    print(d["questions"][s[0]])
