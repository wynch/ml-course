"""Exercise (a) — implement single-head self-attention forward, in numpy.

You are given input token vectors X and the three projection matrices
(W_q, W_k, W_v). Fill in the forward pass of one causal attention head.

No PyTorch. Just numpy. This is the core computation the whole module is built
on; doing it once by hand makes the rest legible.

Run:  uv run python exercises/ex_a_single_head_attention.py
It checks your output against a reference. All-close ⇒ correct.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def single_head_attention(
    X: np.ndarray,      # (T, d_model)  input token vectors
    W_q: np.ndarray,    # (d_model, d_head)
    W_k: np.ndarray,    # (d_model, d_head)
    W_v: np.ndarray,    # (d_model, d_head)
    causal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (output (T, d_head), attention weights (T, T)).

    Steps:
      1. project: Q = X @ W_q, K = X @ W_k, V = X @ W_v      -> each (T, d_head)
      2. scores  = Q @ Kᵀ / sqrt(d_head)                    -> (T, T)
      3. if causal, set scores[i, j] = -inf for j > i (no peeking ahead)
      4. weights = softmax(scores, axis=-1)                 -> rows sum to 1
      5. output  = weights @ V                              -> (T, d_head)
    """
    T, d_model = X.shape
    d_head = W_q.shape[1]

    # TODO(you): 1. compute Q, K, V by projecting X
    Q = ...  # (T, d_head)
    K = ...  # (T, d_head)
    V = ...  # (T, d_head)

    # TODO(you): 2. scaled dot-product scores (T, T)
    scores = ...

    # TODO(you): 3. apply the causal mask (only if `causal`)
    #   hint: build a (T, T) boolean lower-triangular mask with np.tril,
    #   then np.where(mask, scores, -inf)
    if causal:
        scores = ...

    # TODO(you): 4. softmax over the last axis
    weights = ...

    # TODO(you): 5. weighted sum of values
    output = ...

    return output, weights


# --------------------------------------------------------------------------
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

    assert isinstance(out, np.ndarray), "output is still a placeholder — fill in the TODOs"
    assert np.allclose(w.sum(axis=1), 1.0), "attention rows must sum to 1"
    assert np.allclose(np.triu(w, k=1), 0.0), "causal head must not attend to the future"
    assert np.allclose(out, ref_out), "output does not match reference"
    assert np.allclose(w, ref_w), "attention weights do not match reference"
    print("correct: single-head causal attention matches the reference")


if __name__ == "__main__":
    main()
