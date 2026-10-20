"""The yardsticks of this module, on plain numpy arrays.

- ``best_cosines``: for each true vector, the highest cosine with any dictionary column (MMCS is their mean).
- ``seed_match``: one-to-one matching of two dictionaries' columns (Hungarian), and each pair's cosine.
- ``detection``: a rule-based stand-in for auto-interp. Explain a unit by its 3 most common tokens among
  its 50 strongest activations on half A; on half B predict "strongly active" (>= 20% of its half-A maximum)
  iff the current token is one of the 3; balanced accuracy (0.5 = chance).
- ``token_r2``: share of the unit's half-B variance explained by its half-A mean activation per token.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

TOP, N_EXPLAIN, FRAC = 50, 3, 0.2
MIN_A, MIN_POS = 20, 5          # a unit is scored only with 20 firings on half A and 5 strong ones on half B


def unit(m: np.ndarray) -> np.ndarray:
    """Columns scaled to norm 1 (zero columns stay zero)."""
    n = np.linalg.norm(m, axis=0, keepdims=True)
    return m / np.where(n == 0, 1, n)


def best_cosines(truth: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """truth (d, n_true), dec (d, n_dict) -> (n_true,) best cosine of each true column."""
    return (unit(truth).T @ unit(dec)).max(1)


def seed_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b (d, n): cosine of each column of a with its partner in b under a one-to-one matching."""
    c = unit(a).T @ unit(b)
    r, col = linear_sum_assignment(-c)
    out = np.empty(a.shape[1])
    out[r] = c[r, col]
    return out


def explain(a_half: np.ndarray, t_half: np.ndarray, top: int = TOP, n: int = N_EXPLAIN) -> np.ndarray:
    """The n tokens most common among the unit's ``top`` strongest activations (ties: earliest first)."""
    k = min(top, len(a_half))
    idx = np.argpartition(-a_half, k - 1)[:k]
    idx = idx[a_half[idx] > 0]
    toks, first = np.unique(t_half[idx], return_index=True)
    counts = np.array([(t_half[idx] == t).sum() for t in toks])
    order = np.lexsort((first, -counts))
    return toks[order][:n]


def detection(a: np.ndarray, t: np.ndarray):
    """(score, explanation tokens) or (None, None) when the unit fires too rarely to score."""
    h = len(a) // 2
    aA, aB, tA, tB = a[:h], a[h:], t[:h], t[h:]
    if (aA > 0).sum() < MIN_A:
        return None, None
    expl = explain(aA, tA)
    pos = aB >= FRAC * aA.max()
    if pos.sum() < MIN_POS:
        return None, None
    pred = np.isin(tB, expl)
    tpr = (pred & pos).sum() / pos.sum()
    tnr = (~pred & ~pos).sum() / (~pos).sum()
    return float((tpr + tnr) / 2), expl


def token_r2(a: np.ndarray, t: np.ndarray, vocab: int) -> float:
    h = len(a) // 2
    aA, aB, tA, tB = a[:h], a[h:], t[:h], t[h:]
    cnt = np.bincount(tA, minlength=vocab)
    s = np.bincount(tA, weights=aA, minlength=vocab)
    mean = np.where(cnt > 0, s / np.maximum(cnt, 1), aA.mean())
    pred = mean[tB]
    sst = ((aB - aB.mean()) ** 2).sum()
    return float(1 - ((aB - pred) ** 2).sum() / sst) if sst > 0 else float("nan")


def score_units(cols, t: np.ndarray, vocab: int, n_units: int):
    """cols(i) -> dense activation vector of unit i. Returns per-unit detection, token R2, explanation."""
    det, r2, expl = np.full(n_units, np.nan), np.full(n_units, np.nan), [None] * n_units
    for i in range(n_units):
        a = cols(i)
        s, e = detection(a, t)
        if s is None:
            continue
        det[i], expl[i] = s, e
        r2[i] = token_r2(a, t, vocab)
    return det, r2, expl


def summary(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    return {"n": int(len(x)), "median": round(float(np.median(x)), 4), "q25": round(float(np.quantile(x, 0.25)), 4),
            "q75": round(float(np.quantile(x, 0.75)), 4), "mean": round(float(x.mean()), 4),
            "hist": np.histogram(x, bins=20, range=(0, 1))[0].tolist()}
