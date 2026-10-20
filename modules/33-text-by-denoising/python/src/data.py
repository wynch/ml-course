"""Tiny-shakespeare at the character level, as in module 04.

The corpus is module 04's file, copied into ``corpus/``. The tokenizer sorts
the 65 distinct characters; the last 10% of the file is the validation split.
Every number in this module is measured on the same fixed validation blocks,
so two models are always scored on the same text.
"""

from __future__ import annotations

import re

import common
import numpy as np
import torch

BLOCK = 128           # characters per training window and per sampled line


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        """[MASK] (id vocab_size) shows as an underscore, which the corpus never uses."""
        return "".join(self.chars[i] if i < self.vocab_size else "_" for i in ids)


def load():
    """(tokenizer, train ids, val ids) as int64 tensors on the CPU."""
    text = common.CORPUS.read_text()
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = int(len(ids) * 0.1)
    return tok, ids[:-n_val], ids[-n_val:]


def batch(data: torch.Tensor, n: int, gen: torch.Generator, block: int = BLOCK) -> torch.Tensor:
    """n random windows of `block` characters: (n, block). One training "epoch" in this module
    means as many characters drawn as the data holds, not one pass in a fixed order."""
    ix = torch.randint(len(data) - block, (n,), generator=gen)
    return data[ix[:, None] + torch.arange(block)]


def val_blocks(val: torch.Tensor, n: int = 256, block: int = BLOCK) -> torch.Tensor:
    """n non-overlapping validation blocks, evenly spaced over the split: (n, block)."""
    starts = np.linspace(0, len(val) - block, n).astype(int)
    return torch.stack([val[s: s + block] for s in starts])


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def word_set(train: torch.Tensor, tok: CharTokenizer) -> set[str]:
    """Every word of the training split, lower-cased: the dictionary for the word check."""
    return {w.lower() for w in words(tok.decode(train.tolist()))}


def word_score(text: str, vocab: set[str]) -> float:
    """Share of the complete words in `text` found in the training split. The first and last
    word of a line are cut by the window, so they are not counted."""
    ws = words(text)[1:-1]
    return float(np.mean([w.lower() in vocab for w in ws])) if ws else 0.0


def log_unigram(train: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Log frequency of each character in the training split (vocab_size,)."""
    c = torch.bincount(train, minlength=vocab_size).double()
    return (c / c.sum()).log().float()
