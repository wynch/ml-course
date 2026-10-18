"""Exact match and F1, as in the official SQuAD v1.1 evaluation script.

Normalise both strings: lowercase, drop punctuation, drop the articles a, an,
the, collapse whitespace. Exact match is 1 if the normalised strings are
equal. F1 is the harmonic mean of token precision and recall, counting each
shared token as often as it appears in both. A prediction is scored against
every gold answer and keeps the best.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def best(pred: str, golds: list[str]) -> tuple[float, float]:
    """(exact match, F1), each the max over the gold answers."""
    return max(exact(pred, g) for g in golds), max(f1(pred, g) for g in golds)
