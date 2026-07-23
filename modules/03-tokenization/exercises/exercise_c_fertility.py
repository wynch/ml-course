"""Exercise (c) — measure and plot SmolLM3's fertility on a language you pick.

"Fertility" = average number of tokens the tokenizer spends per whitespace
word. English tends to be cheap (SmolLM3 was trained on lots of it); other
languages pay a tax. Pick a language, gather a handful of sentences, and see
how badly SmolLM3 over-tokenizes it versus English.

    uv run pytest tests/test_solutions.py -k fertility
    uv run python exercises/exercise_c_fertility.py    # writes a PNG

Reference solution: ../solutions/solution_c_fertility.py.
"""

from __future__ import annotations


def measure_fertility(tokenizer, sentences: list[str]) -> float:
    """Return mean tokens-per-word over ``sentences`` using ``tokenizer``.

    ``tokenizer`` is a Hugging Face tokenizer, so ``tokenizer.tokenize(s)``
    gives the list of token strings for sentence ``s``. A "word" is a
    whitespace-separated chunk.
    """
    # TODO(you): sum tokens and words across all sentences and return the ratio
    # (total tokens / total words). Guard against division by zero.
    raise NotImplementedError("implement measure_fertility")


# TODO(you): replace/extend these with sentences in the language of YOUR choice.
MY_LANGUAGE = "spanish"
MY_SENTENCES = [
    "El veloz murciélago hindú comía feliz cardillo y kiwi.",
    "La cigüeña tocaba el saxofón detrás del palenque de paja.",
]

ENGLISH_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
]


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    en = measure_fertility(tok, ENGLISH_SENTENCES)
    mine = measure_fertility(tok, MY_SENTENCES)
    print(f"english fertility : {en:.2f} tokens/word")
    print(f"{MY_LANGUAGE} fertility : {mine:.2f} tokens/word")
    print(f"{MY_LANGUAGE} pays a {mine / en:.2f}x tax vs english")
