"""Verify the reference solutions for exercises (a), (b), (c)."""

import os

import pytest
from bpe import BPETokenizer

HERE = os.path.dirname(__file__)
CORPUS = os.path.join(HERE, "..", "..", "corpus", "input.txt")


@pytest.fixture(scope="module")
def tok():
    data = open(CORPUS, "rb").read()[:100_000]
    return BPETokenizer.train(data, 100)


# ---- (a) merge --------------------------------------------------------------
def test_merge_solution():
    from solution_a_merge import merge_apply

    assert merge_apply([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert merge_apply([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_apply([5, 6, 7], (0, 0), 9) == [5, 6, 7]
    assert merge_apply([], (1, 2), 9) == []


# ---- (b) special tokens -----------------------------------------------------
def test_special_token_round_trip(tok):
    from solution_b_specials import ENDOFTEXT, decode_with_specials, encode_with_specials

    text = f"hello{ENDOFTEXT}world{ENDOFTEXT}again"
    ids = encode_with_specials(tok, text)
    special_id = tok.specials[ENDOFTEXT]
    assert ids.count(special_id) == 2
    assert decode_with_specials(tok, ids) == text


def test_special_token_is_atomic(tok):
    from solution_b_specials import ENDOFTEXT, encode_with_specials

    # the special must be ONE id, not the bytes of "<|endoftext|>"
    ids = encode_with_specials(tok, ENDOFTEXT)
    assert ids == [tok.specials[ENDOFTEXT]]


# ---- (c) fertility ----------------------------------------------------------
def test_measure_fertility_basic():
    from solution_c_fertility import measure_fertility

    class FakeTok:
        # 1 token per character -> fertility == avg chars per word
        def tokenize(self, s):
            return list(s.replace(" ", ""))

    # "ab cd" -> 4 tokens / 2 words = 2.0
    assert measure_fertility(FakeTok(), ["ab cd"]) == 2.0


def test_fertility_english_cheaper_than_spanish():
    from solution_c_fertility import run

    res = run(save=False)
    assert res["english"] > 0 and res["spanish"] > 0
    # SmolLM3 is English-heavy: Spanish should cost at least as much per word.
    assert res["spanish"] >= res["english"]
