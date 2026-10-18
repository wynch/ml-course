"""Solution (b): SQuAD's answer normalisation, exact match and F1."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    golds = ["Denver Broncos", "the Denver Broncos"]
    for pred in ("Denver Broncos", "The Denver Broncos!", "Broncos", "The Denver Broncos won Super Bowl 50", "Panthers"):
        em = max(exact(pred, g) for g in golds)
        f = max(f1(pred, g) for g in golds)
        print(f"{pred!r:42s} EM {em:.0f}  F1 {f:.3f}")


if __name__ == "__main__":
    main()
