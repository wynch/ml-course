"""Solution (b): explanation, detection score and token R2."""

from __future__ import annotations

import numpy as np


def explain(a_half: np.ndarray, t_half: np.ndarray, top: int = 50, n: int = 3) -> np.ndarray:
    k = min(top, len(a_half))
    idx = np.argpartition(-a_half, k - 1)[:k]
    idx = idx[a_half[idx] > 0]
    toks, first = np.unique(t_half[idx], return_index=True)
    counts = np.array([(t_half[idx] == x).sum() for x in toks])
    return toks[np.lexsort((first, -counts))][:n]


def detection(a: np.ndarray, t: np.ndarray):
    h = len(a) // 2
    aA, aB, tA, tB = a[:h], a[h:], t[:h], t[h:]
    if (aA > 0).sum() < 20:
        return None, None
    e = explain(aA, tA)
    pos = aB >= 0.2 * aA.max()
    if pos.sum() < 5:
        return None, None
    pred = np.isin(tB, e)
    return float(((pred & pos).sum() / pos.sum() + (~pred & ~pos).sum() / (~pos).sum()) / 2), e


def token_r2(a: np.ndarray, t: np.ndarray, vocab: int) -> float:
    h = len(a) // 2
    aA, aB, tA, tB = a[:h], a[h:], t[:h], t[h:]
    cnt = np.bincount(tA, minlength=vocab)
    mean = np.where(cnt > 0, np.bincount(tA, weights=aA, minlength=vocab) / np.maximum(cnt, 1), aA.mean())
    pred = mean[tB]
    return float(1 - ((aB - pred) ** 2).sum() / ((aB - aB.mean()) ** 2).sum())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = rng.integers(0, 50, size=20000)
    one_token = np.where(t == 7, 1.0, 0.0)
    sad = rng.random(20000) < 0.5
    context = np.where((t == 7) & sad, 1.0, 0.0)
    # Detection stays near 1 for the context unit: its wrong guesses fall on one rare token, a tiny share
    # of the negatives. Token R2 falls to about 0.45: the token gives the mean, not which half fires.
    for name, a in (("fires on token 7", one_token), ("token 7, half the time", context)):
        s, e = detection(a, t)
        print(f"{name}: detection {s:.3f} (explanation {e}), token R2 {token_r2(a, t, 50):.3f}")
