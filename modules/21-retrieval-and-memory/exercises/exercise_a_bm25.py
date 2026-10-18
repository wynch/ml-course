"""Exercise (a): an inverted index and the BM25 score over it.

1. ``build_index``: for each term, the list of (document, count) pairs where it
   occurs, in document order; and each document's length in tokens.
2. ``idf``: ln(1 + (N - df + 0.5) / (df + 0.5)).
3. ``bm25_scores``: for each query term, walk only its postings and add
   idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl)).

Predict before you run: on the four toy documents in ``main``, which has the
larger idf, "the" or "market"? And does the query "cats" score the first
document, which says "cat"?

Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_bm25.py
    uv run pytest tests/test_solutions.py -k bm25 -v
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from bm25 import tokenize  # noqa: E402

K1, B = 1.5, 0.75


def build_index(docs: list[str]) -> tuple[dict[str, list[tuple[int, int]]], list[int]]:
    """term -> [(doc, tf), ...] in doc order, and each document's length in tokens."""
    postings: dict[str, list[tuple[int, int]]] = {}
    lengths = []
    for i, d in enumerate(docs):
        toks = tokenize(d)
        # TODO(you): append len(toks) to lengths; for each (term, tf) in Counter(toks),
        # append (i, tf) to postings[term].
        ...
    return postings, lengths


def idf(n: int, df: int) -> float:
    # TODO(you): one line.
    ...


def bm25_scores(postings, lengths, query: str, k1: float = K1, b: float = B) -> list[float]:
    n = len(lengths)
    avgdl = sum(lengths) / n
    out = [0.0] * n
    for term in tokenize(query):
        # TODO(you): for each (doc, tf) in this term's postings, add its BM25 part to out[doc].
        ...
    return out


def main() -> None:
    docs = ["The cat sat on the mat.", "A dog chased the cat.", "Dogs and cats, cats and dogs.",
            "The stock market fell."]
    postings, lengths = build_index(docs)
    print(f"postings for 'cat': {postings['cat']}; idf('the') = {idf(len(docs), len(postings['the'])):.3f}, "
          f"idf('market') = {idf(len(docs), len(postings['market'])):.3f}")
    for q in ("cat", "the cat", "cats"):
        s = bm25_scores(postings, lengths, q)
        print(f"{q!r:10s} -> " + "  ".join(f"{x:.3f}" for x in s))


if __name__ == "__main__":
    main()
