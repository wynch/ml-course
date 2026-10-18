"""Exercise (b): SQuAD's answer normalisation, exact match and F1.

1. ``normalize``: lowercase, drop every character in string.punctuation, drop
   the words a, an, the, collapse runs of whitespace to one space.
2. ``exact``: 1.0 if the normalised strings are equal, else 0.0.
3. ``f1``: split both normalised strings on spaces; count the tokens they
   share (each as often as it appears in both: Counter & Counter); precision is
   shared / predicted tokens, recall is shared / gold tokens; F1 is their
   harmonic mean, and 0 if nothing is shared.

Predict before you run: what are the exact match and the F1 of the reply
"The Denver Broncos won Super Bowl 50" against the gold answer "Denver
Broncos"?

Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_squad_f1.py
    uv run pytest tests/test_solutions.py -k squad -v
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    # TODO(you): drop punctuation, then the articles (re.sub with \b), then collapse whitespace.
    ...


def exact(pred: str, gold: str) -> float:
    # TODO(you): one line.
    ...


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    # TODO(you): shared count, precision, recall, harmonic mean (0.0 if nothing shared).
    ...


def main() -> None:
    golds = ["Denver Broncos", "the Denver Broncos"]
    for pred in ("Denver Broncos", "The Denver Broncos!", "Broncos", "The Denver Broncos won Super Bowl 50", "Panthers"):
        em = max(exact(pred, g) for g in golds)
        f = max(f1(pred, g) for g in golds)
        print(f"{pred!r:42s} EM {em:.0f}  F1 {f:.3f}")


if __name__ == "__main__":
    main()
