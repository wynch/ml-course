"""Embedding geometry: the input embedding matrix is a learned map of meaning.

Row *i* of ``embed_tokens`` is the 960-dim vector the model starts from when it
sees token *i*. We project a hand-picked set of tokens down to 2D with PCA and
watch related tokens cluster — numbers, months, countries, code keywords — then
do a nearest-neighbour lookup in the full 960-dim space.

Run me:
    uv run python src/embeddings.py            # print nearest neighbours
    uv run python src/embeddings.py --figure   # draw the 2D PCA projection
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_model, load_tokenizer

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"

# Groups chosen so each is a clear semantic family. We prepend a space because
# SmolLM2's BPE encodes most mid-sentence words with a leading space.
GROUPS = {
    "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight"],
    "months": ["January", "February", "March", "April", "June", "July", "October"],
    "countries": ["France", "Germany", "Spain", "Japan", "China", "Brazil", "Egypt"],
    "code": ["def", "return", "import", "class", "while", "print", "None"],
    "colors": ["red", "green", "blue", "yellow", "purple", "orange"],
}
GROUP_COLORS = {
    "numbers": "#4C72B0",
    "months": "#DD8452",
    "countries": "#55A868",
    "code": "#C44E52",
    "colors": "#8172B3",
}


def _single_token_id(tokenizer, word: str) -> int | None:
    """Return the token id if ``' word'`` encodes to exactly one token, else None."""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def collect_embeddings(model, tokenizer):
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    labels, vectors, colors, groups = [], [], [], []
    for group, words in GROUPS.items():
        for w in words:
            tid = _single_token_id(tokenizer, w)
            if tid is None:
                continue
            labels.append(w)
            vectors.append(emb[tid])
            colors.append(GROUP_COLORS[group])
            groups.append(group)
    return labels, torch.stack(vectors), colors, groups, emb


def nearest_neighbours(word: str, model, tokenizer, k: int = 6):
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    tid = _single_token_id(tokenizer, word)
    if tid is None:
        return None
    q = F.normalize(emb[tid : tid + 1], dim=-1)
    alln = F.normalize(emb, dim=-1)
    sims = (alln @ q.T).squeeze(-1)
    sims[tid] = -1  # exclude self
    vals, idx = sims.topk(k)
    return [(tokenizer.decode([i]).strip(), float(v)) for i, v in zip(idx.tolist(), vals.tolist())]


def fig_pca(model, tokenizer, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    labels, vectors, colors, groups, _ = collect_embeddings(model, tokenizer)
    coords = PCA(n_components=2, random_state=0).fit_transform(vectors.numpy())

    fig, ax = plt.subplots(figsize=(9, 7))
    seen = set()
    for (x, y), lbl, c, g in zip(coords, labels, colors, groups):
        ax.scatter(x, y, c=c, s=90, edgecolors="white", linewidths=0.8, zorder=3,
                   label=g if g not in seen else None)
        seen.add(g)
        ax.annotate(lbl, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.legend(title="token family", loc="best", framealpha=0.9)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(
        "Input embeddings (960-dim) → PCA 2D\nsemantic families cluster before a single layer runs",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    model = load_model()
    tok = load_tokenizer()

    print("nearest neighbours in embedding space (cosine):")
    for w in ["three", "July", "France", "def", "blue"]:
        nn = nearest_neighbours(w, model, tok)
        if nn:
            shown = ", ".join(f"{t}({s:.2f})" for t, s in nn)
            print(f"  {w:>8} -> {shown}")

    if args.figure:
        FIG_DIR.mkdir(exist_ok=True)
        fig_pca(model, tok, FIG_DIR / "embedding_pca.png")


if __name__ == "__main__":
    main()
