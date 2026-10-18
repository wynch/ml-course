"""Exercise (c): MiniLM by hand with transformers, and a cosine search.

The sentence-transformers library wraps three steps that this exercise writes
out: run the BERT body, average its last hidden states over the real tokens
(not the padding), and scale each vector to length 1. Then a cosine search is
one matrix product.

Predict before you run: the query "Where did the pet rest?" shares no word
with "A kitten slept on the rug." Which of the four documents comes first?
Which would BM25 put first?

Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_c_embed_search.py
    uv run pytest tests/test_solutions.py -k embed -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: E402  (MPS guard before torch)
import torch  # noqa: E402


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the real tokens: hidden (batch, seq, d), mask (batch, seq) of 0 and 1."""
    # TODO(you): zero the padded positions, sum over seq, divide by the number of real tokens.
    ...


@torch.inference_mode()
def embed(model, tok, texts: list[str]) -> torch.Tensor:
    """Unit vectors (len(texts), 384): tokenise with padding, run the model, mean-pool, L2-normalise."""
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    # TODO(you): mean_pool, then torch.nn.functional.normalize along the last dimension.
    ...


def search(q: torch.Tensor, P: torch.Tensor, k: int) -> list[tuple[int, float]]:
    """The k rows of P with the largest cosine to q (all unit vectors), best first."""
    # TODO(you): one matrix-vector product and torch.topk.
    ...


def main() -> None:
    model, tok = common.load_embedder("cpu")
    docs = ["The cat sat on the mat.", "Interest rates rose for the third month in a row.",
            "A kitten slept on the rug.", "The central bank raised its key rate again."]
    P = embed(model, tok, docs)
    for query in ("Where did the pet rest?", "What did the bank do with borrowing costs?"):
        hits = search(embed(model, tok, [query])[0], P, 2)
        print(query, "->", [(docs[i], round(s, 3)) for i, s in hits])


if __name__ == "__main__":
    main()
