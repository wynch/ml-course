"""Solution (c): a few-shot prompt and label-word scoring, by hand.

Build the prompt the module uses, run SmolLM2-360M-Instruct on it once, and read the
answer from two next-token logits. Runs on the CPU in a few seconds once the model is
in the Hugging Face cache.

    uv run python ../solutions/solution_c_label_words.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common
import torch

HEADER = "Classify the sentiment of each movie review as positive or negative.\n\n"
LABELS = {0: "negative", 1: "positive"}


def format_prompt(test: str, demos: list[tuple[str, int]]) -> str:
    shots = "".join(f"Review: {s}\nSentiment: {LABELS[y]}\n\n" for s, y in demos)
    return f"{HEADER}{shots}Review: {test}\nSentiment:"


def label_ids(tok) -> tuple[int, int]:
    """Token ids of " negative" and " positive" (the leading space matters)."""
    (neg,), (pos,) = tok.encode(" negative"), tok.encode(" positive")
    return neg, pos


@torch.no_grad()
def margin(model, tok, prompt: str) -> float:
    """logit(" positive") - logit(" negative") for the token after the prompt."""
    neg, pos = label_ids(tok)
    logits = model(**tok(prompt, return_tensors="pt")).logits[0, -1]
    return float(logits[pos] - logits[neg])


def main() -> None:
    tok, model = common.load_tokenizer(), common.load_hf_model("cpu")
    demos = [("a gorgeous , witty , seductive movie .", 1), ("the plot is nothing but boilerplate clichés .", 0)]
    for test in ("an utterly charming and hilarious film", "a dull , lifeless mess"):
        for k in (0, 2):
            m = margin(model, tok, format_prompt(test, demos[:k]))
            print(f"k = {k}  {test!r}: margin {m:+.2f} -> {'positive' if m > 0 else 'negative'}")


if __name__ == "__main__":
    main()
