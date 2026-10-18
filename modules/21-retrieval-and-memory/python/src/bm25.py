"""BM25 over an inverted index, by hand.

The tokeniser is deliberately plain so the explorable can repeat it in a few
lines of JavaScript: Unicode NFKD, drop combining accents, lowercase, keep runs
of [a-z0-9]. No stemming, no stop words; BM25's idf already gives "the" a
weight near zero.

For a query q and a document d, with N documents, df(t) documents holding
term t, tf(t, d) its count in d, |d| the document's length in tokens and
avgdl the mean length:

    idf(t)    = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score     = sum over t in q of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

This is the Lucene form of the idf (always positive). k1 = 1.5 and b = 0.75
are the usual defaults; we did not tune them. A query term counts once per
time it appears in the query.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

K1 = 1.5
B = 0.75
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return _WORD.findall(t.lower())


class BM25:
    """An inverted index (term -> list of (doc, tf)) and the BM25 score over it."""

    def __init__(self, docs: list[str], k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        for i, d in enumerate(docs):
            toks = tokenize(d)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.n = len(docs)
        self.avgdl = sum(self.doc_len) / self.n
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> list[float]:
        """Score every document; only the postings of the query's terms are read."""
        out = [0.0] * self.n
        for term in tokenize(query):
            for doc, tf in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc] / self.avgdl)
                out[doc] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out

    def postings_read(self, query: str) -> int:
        return sum(len(self.postings.get(t, ())) for t in tokenize(query))

    def top(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        order = sorted(range(self.n), key=lambda i: (-s[i], i))[:k]
        return [(i, s[i]) for i in order]


class Scan:
    """The same score with no index: every query visits every document. For tests and timing."""

    def __init__(self, docs: list[str], k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.doc_len = [sum(c.values()) for c in self.docs]
        self.n = len(docs)
        self.avgdl = sum(self.doc_len) / self.n

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        df = {t: sum(1 for c in self.docs if t in c) for t in set(terms)}
        out = []
        for c, length in zip(self.docs, self.doc_len):
            s = 0.0
            for t in terms:
                tf = c.get(t, 0)
                if tf:
                    idf = math.log(1 + (self.n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avgdl))
            out.append(s)
        return out
