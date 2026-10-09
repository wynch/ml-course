"""Exercise (b) — add special-token handling to encode / decode.

Real tokenizers reserve ids for control tokens like ``<|endoftext|>`` that
must never be produced by merging ordinary bytes. The document boundary is a
single atomic token, not the literal characters ``<``, ``|``, ``e`` ...

Your job: implement `encode_with_specials` and `decode_with_specials` so that
the special string round-trips as ONE reserved id.

    uv run pytest tests/test_solutions.py -k special

Reference solution: ../solutions/solution_b_specials.py.
"""

from __future__ import annotations

from bpe import BPETokenizer

ENDOFTEXT = "<|endoftext|>"


def encode_with_specials(tok: BPETokenizer, text: str, special: str = ENDOFTEXT) -> list[int]:
    """Encode ``text``, emitting a single reserved id wherever ``special`` occurs.

    Steps:
      1. make sure ``special`` is registered (tok.add_special returns its id)
      2. split ``text`` on the literal ``special`` string
      3. BPE-encode each chunk, and join the chunks with the special id between
    """
    # TODO(you): register the special token, split the text on it, encode each
    # piece with tok.encode(...), and interleave the special id between pieces.
    raise NotImplementedError("implement encode_with_specials")


def decode_with_specials(tok: BPETokenizer, ids: list[int], special: str = ENDOFTEXT) -> str:
    """Decode ids back to text. The reserved id becomes ``special`` again.

    Note: ``BPETokenizer.decode`` already restores any id registered via
    ``add_special`` — so once the id is registered this can be a one-liner.
    """
    # TODO(you): return the decoded string (hint: tok.decode already knows how
    # to turn a registered special id back into its surface form).
    raise NotImplementedError("implement decode_with_specials")


if __name__ == "__main__":
    data = open("../corpus/input.txt", "rb").read()[:100_000]
    tok = BPETokenizer.train(data, 100)
    ids = encode_with_specials(tok, f"hello{ENDOFTEXT}world")
    special_id = tok.specials[ENDOFTEXT]
    assert ids.count(special_id) == 1
    assert decode_with_specials(tok, ids) == f"hello{ENDOFTEXT}world"
    print("special-token handling works!")
