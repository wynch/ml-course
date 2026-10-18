"""Exercise (c): a few-shot prompt and label-word scoring, by hand.

Build the prompt the module uses, run SmolLM2-360M-Instruct on it once, and read the
answer from two next-token logits; nothing is generated.

1. ``format_prompt``: the header, then one block per example
   ("Review: <text>\nSentiment: <label>\n\n"), then "Review: <test>\nSentiment:".
2. ``label_ids``: the token ids of " negative" and " positive" (with the space).
3. ``margin``: logit(" positive") - logit(" negative") at the last position.

Predict before you run: does the sign of the margin change between 0 and 2 examples
for these two clear reviews? Runs on the CPU in a few seconds once the model is cached.
Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_c_label_words.py
    uv run pytest tests/test_solutions.py -k label -v
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
    # TODO(you): join the example blocks, then add the test block with its label left open.
    ...


def label_ids(tok) -> tuple[int, int]:
    """Token ids of " negative" and " positive" (the leading space matters)."""
    # TODO(you): tok.encode each word; check each gives one id.
    ...


@torch.no_grad()
def margin(model, tok, prompt: str) -> float:
    """logit(" positive") - logit(" negative") for the token after the prompt."""
    neg, pos = label_ids(tok)
    # TODO(you): one forward pass; take the logits of the last position.
    ...


def main() -> None:
    tok, model = common.load_tokenizer(), common.load_hf_model("cpu")
    demos = [("a gorgeous , witty , seductive movie .", 1), ("the plot is nothing but boilerplate clichés .", 0)]
    for test in ("an utterly charming and hilarious film", "a dull , lifeless mess"):
        for k in (0, 2):
            m = margin(model, tok, format_prompt(test, demos[:k]))
            print(f"k = {k}  {test!r}: margin {m:+.2f} -> {'positive' if m > 0 else 'negative'}")


if __name__ == "__main__":
    main()
