"""The task: add two 3-digit numbers, and the program that checks the answer.

A prompt is ``"374+485="``. A reply ends with ``"."`` and comes in one of two
styles, both in the training data of the base model:

- direct: ``"859."``
- worked: ``"[4+5=9,7+8+0=15,3+4+1=8]859."``, column by column with the carry, then the sum.

The checker reads the text after the last ``"]"`` (or the whole reply when
there is none) and gives 1 if it is exactly the sum and the reply ended,
0 otherwise. The worked steps are not graded: only the final number counts.
"""

from __future__ import annotations

import math

import numpy as np

CHARS = "0123456789+=[],._"          # "_" pads; it is never a target
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
PAD = STOI["_"]
END = STOI["."]
DIGITS = 3
LO, HI = 10 ** (DIGITS - 1), 10 ** DIGITS - 1     # both numbers in 100..999
PROMPT_LEN = 2 * DIGITS + 2                       # "374+485="
MAX_REPLY = 32                                    # the longest worked reply is 31 characters
N_TEST = 500


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(CHARS[int(i)] for i in ids)


def prompt(a: int, b: int) -> str:
    return f"{a}+{b}="


def direct(a: int, b: int) -> str:
    return f"{a + b}."


def worked(a: int, b: int) -> str:
    """Column by column from the right: digit + digit (+ carry) = column sum."""
    steps, c = [], 0
    for i in range(DIGITS):
        x, y = a // 10 ** i % 10, b // 10 ** i % 10
        s = x + y + c
        steps.append(f"{x}+{y}={s}" if i == 0 else f"{x}+{y}+{c}={s}")
        c = s // 10
    return "[" + ",".join(steps) + f"]{a + b}."


def reply_text(ids) -> str:
    """Characters of a sampled reply up to and including the first END; the rest is dropped."""
    s = decode(ids)
    i = s.find(".")
    return s if i < 0 else s[: i + 1]


def check(reply: str, a: int, b: int) -> int:
    """1 if the reply ends and its final number is a + b, else 0."""
    if not reply.endswith("."):
        return 0
    final = reply[:-1].rsplit("]", 1)[-1]
    return int(final == str(a + b))


def is_worked(reply: str) -> bool:
    return reply.startswith("[")


def splits(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """500 test pairs drawn once; training draws any other pair.

    Returns (test pairs, a function that draws n training pairs from a numpy rng).
    """
    rng = np.random.default_rng(seed)
    test = rng.integers(LO, HI + 1, size=(N_TEST, 2))
    held = {(int(a), int(b)) for a, b in test}

    def draw(n: int, r: np.random.Generator) -> np.ndarray:
        out = r.integers(LO, HI + 1, size=(n, 2))
        for i in range(n):
            while (int(out[i, 0]), int(out[i, 1])) in held:
                out[i] = r.integers(LO, HI + 1, size=2)
        return out

    return test, draw


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k samples, drawn without replacement from n with c right, is right.

    The unbiased estimator of Chen et al. (2021): 1 - C(n-c, k) / C(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
