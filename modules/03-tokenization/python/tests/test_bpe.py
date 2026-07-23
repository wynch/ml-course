"""Correctness of the from-scratch byte-level BPE."""

import os

import pytest
from bpe import BPETokenizer, best_pair, get_stats, merge

HERE = os.path.dirname(__file__)
CORPUS = os.path.join(HERE, "..", "..", "corpus", "input.txt")


@pytest.fixture(scope="module")
def tok():
    data = open(CORPUS, "rb").read()[:200_000]  # a slice keeps the test quick
    return BPETokenizer.train(data, 50)


def test_get_stats_counts_pairs():
    assert get_stats([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}


def test_merge_is_greedy_nonoverlapping():
    # [1,1,1] with pair (1,1)->9 merges the first pair only, leaving one 1.
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge([0, 1, 2, 1, 2], (1, 2), 7) == [0, 7, 7]


def test_best_pair_tie_break_is_smallest():
    # (0,1) and (2,3) both appear twice; the smaller pair (0,1) must win.
    counts = {(2, 3): 2, (0, 1): 2, (9, 9): 1}
    assert best_pair(counts) == (0, 1)


def test_vocab_size_accounting(tok):
    assert tok.vocab_size == 256 + len(tok.merges)
    assert len(tok.merges) == 50


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "The quick brown fox.",
        "To be, or not to be: that is the question.",
        "unicode: café, naïve, Noël, 日本語, 🚀",
        "numbers 4096 and symbols @#$%^&*()",
    ],
)
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_on_corpus_slice(tok):
    blob = open(CORPUS, encoding="utf-8").read()[:10_000]
    assert tok.decode(tok.encode(blob)) == blob


def test_encoding_actually_compresses(tok):
    text = open(CORPUS, "rb").read()[:20_000]
    assert len(tok.encode_bytes(text)) < len(text)


def test_from_merges_matches_full_train(tok):
    rebuilt = BPETokenizer.from_merges(tok.merges)
    assert rebuilt.merges == tok.merges
    assert rebuilt.encode("hello world") == tok.encode("hello world")
