"""Char-level dataset for tiny-shakespeare.

The whole "tokenizer" here is: sort the unique characters, map each to an int.
That is deliberately the dumbest thing that works — module 03 is where real
sub-word tokenization lives. Here we want the transformer, not the tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CORPUS = Path(__file__).resolve().parents[2] / "corpus" / "tiny_shakespeare.txt"


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        # store the ordered vocab; index == token id
        chars = [self.itos[i] for i in range(self.vocab_size)]
        path.write_text(json.dumps({"chars": chars}, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        chars = json.loads(path.read_text())["chars"]
        obj = cls.__new__(cls)
        obj.itos = {i: c for i, c in enumerate(chars)}
        obj.stoi = {c: i for i, c in enumerate(chars)}
        obj.vocab_size = len(chars)
        return obj


def load_corpus() -> str:
    return CORPUS.read_text()


def make_splits(text: str, tok: CharTokenizer, val_frac: float = 0.1):
    data = np.array(tok.encode(text), dtype=np.int64)
    n_val = int(len(data) * val_frac)
    train = torch.from_numpy(data[:-n_val])
    val = torch.from_numpy(data[-n_val:])
    return train, val


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
