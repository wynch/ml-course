"""Solution (a) — single-head causal self-attention in numpy."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def single_head_attention(X, W_q, W_k, W_v, causal: bool = True):
    T, d_model = X.shape
    d_head = W_q.shape[1]

    # 1. project into queries, keys, values
    Q = X @ W_q          # (T, d_head)
    K = X @ W_k          # (T, d_head)
    V = X @ W_v          # (T, d_head)

    # 2. scaled dot-product scores
    scores = Q @ K.T / np.sqrt(d_head)          # (T, T)

    # 3. causal mask: position i may only read positions j <= i
    if causal:
        mask = np.tril(np.ones((T, T), dtype=bool))
        scores = np.where(mask, scores, -np.inf)

    # 4. softmax -> attention weights (each row sums to 1)
    weights = softmax(scores, axis=-1)

    # 5. weighted average of the values
    output = weights @ V                         # (T, d_head)
    return output, weights


def _reference(X, W_q, W_k, W_v, causal=True):
    d_head = W_q.shape[1]
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    scores = Q @ K.T / np.sqrt(d_head)
    if causal:
        mask = np.tril(np.ones(scores.shape, dtype=bool))
        scores = np.where(mask, scores, -np.inf)
    w = softmax(scores, axis=-1)
    return w @ V, w


def main() -> None:
    rng = np.random.default_rng(42)
    T, d_model, d_head = 5, 12, 4
    X = rng.standard_normal((T, d_model))
    W_q = rng.standard_normal((d_model, d_head))
    W_k = rng.standard_normal((d_model, d_head))
    W_v = rng.standard_normal((d_model, d_head))

    out, w = single_head_attention(X, W_q, W_k, W_v, causal=True)
    ref_out, ref_w = _reference(X, W_q, W_k, W_v, causal=True)

    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(np.triu(w, k=1), 0.0)
    assert np.allclose(out, ref_out)
    assert np.allclose(w, ref_w)
    print("correct: single-head causal attention matches the reference")


if __name__ == "__main__":
    main()
