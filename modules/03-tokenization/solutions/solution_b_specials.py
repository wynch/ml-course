"""Solution (b) — special-token handling for encode / decode."""

from __future__ import annotations

from bpe import BPETokenizer

ENDOFTEXT = "<|endoftext|>"


def encode_with_specials(tok: BPETokenizer, text: str, special: str = ENDOFTEXT) -> list[int]:
    special_id = tok.add_special(special)  # idempotent: returns existing id if set
    chunks = text.split(special)
    ids: list[int] = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            ids.append(special_id)  # boundary between chunks
        ids.extend(tok.encode(chunk))
    return ids


def decode_with_specials(tok: BPETokenizer, ids: list[int], special: str = ENDOFTEXT) -> str:
    # BPETokenizer.decode already maps registered special ids back to their text.
    tok.add_special(special)
    return tok.decode(ids)


if __name__ == "__main__":
    data = open("../corpus/input.txt", "rb").read()[:100_000]
    tok = BPETokenizer.train(data, 100)
    ids = encode_with_specials(tok, f"hello{ENDOFTEXT}world")
    assert ids.count(tok.specials[ENDOFTEXT]) == 1
    assert decode_with_specials(tok, ids) == f"hello{ENDOFTEXT}world"
    print("ok")
