"""Step 1 — scaled dot-product attention, in numpy, on a toy sequence.

Attention is a *soft dictionary lookup*. Each position emits a query; every
position emits a key and a value. The query is compared against all keys
(dot-product), the scores are scaled and softmaxed into weights, and the output
is the weighted average of the values. Causal masking just forbids a position
from looking at the future before the softmax.

This script computes the QKᵀ score matrix for a 6-token toy sequence and draws
it four ways — raw scores, causally masked, softmax of raw, softmax of masked —
so you can see exactly what the mask does.

Run:  uv run python scripts/step1_numpy_attention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

TOKENS = ["The", "cat", "sat", "on", "the", "mat"]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q,K,V: (T, d).  Returns (output (T, d), attention weights (T, T))."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (T, T)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)       # each row sums to 1
    return weights @ V, weights, scores


def annotate(ax, M, title, cmap, fmt="{:.2f}", masked=False):
    # -inf entries (masked-out future) display as blank cells
    disp = np.where(np.isinf(M), np.nan, M)
    im = ax.imshow(disp, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(TOKENS)))
    ax.set_yticks(range(len(TOKENS)))
    ax.set_xticklabels(TOKENS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(TOKENS, fontsize=8)
    ax.set_xlabel("key (attended to)", fontsize=8)
    ax.set_ylabel("query (attending)", fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isinf(v):
                ax.text(j, i, "-∞", ha="center", va="center", fontsize=7, color="#888")
            else:
                shade = "white" if (not np.isnan(disp[i, j]) and abs(v) > 0.6 * np.nanmax(np.abs(disp))) else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color=shade)
    return im


def main() -> None:
    rng = np.random.default_rng(0)
    T, d = len(TOKENS), 8
    # random but fixed Q/K/V for the toy sequence
    Q = rng.standard_normal((T, d))
    K = rng.standard_normal((T, d))
    V = rng.standard_normal((T, d))

    causal = np.tril(np.ones((T, T), dtype=bool))   # True where allowed

    _, w_full, scores_full = scaled_dot_product_attention(Q, K, V)
    _, w_causal, scores_causal = scaled_dot_product_attention(Q, K, V, mask=causal)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    annotate(axes[0, 0], scores_full, "1. raw scores  QKᵀ/√d", "coolwarm")
    annotate(axes[0, 1], scores_causal, "2. after causal mask", "coolwarm", masked=True)
    annotate(axes[1, 0], w_full, "3. softmax(raw)  — sees future", "viridis")
    annotate(axes[1, 1], w_causal, "4. softmax(masked)  — causal", "viridis")
    fig.suptitle(
        "Scaled dot-product attention on a toy sequence\n"
        "row = who is asking (query), column = who is being read (key)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = C.FIGURES / "step1_sdpa_masking.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    # sanity: causal softmax rows are lower-triangular and sum to 1
    assert np.allclose(w_causal.sum(axis=1), 1.0)
    assert np.allclose(np.triu(w_causal, k=1), 0.0), "future must have zero weight"
    print("checks passed: causal rows sum to 1 and never attend to the future")


if __name__ == "__main__":
    main()
