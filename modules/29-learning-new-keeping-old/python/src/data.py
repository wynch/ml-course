"""The two texts, their shared character vocabulary, batches and the fixed
evaluation windows.

Old text (A): tiny Shakespeare, module 04's corpus, copied into
`../corpus/`; the last 10% is held out, as in module 04.
New text (B): GSM8K's worked solutions, as "Q: ...\\nA: ...\\n\\n"; training
from GSM8K's train file, held out from its test file.

A few typographic characters in GSM8K are mapped to ASCII; any other
character outside ASCII is dropped. The vocabulary covers both texts from the
start, so the model trained on A already has a row for every character of B.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from common import CORPUS, GSM8K, parquet

BLOCK = 128
MAP = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-", "×": "*", "÷": "/",
       "\xa0": " ", "\t": " "}
N_EVAL_WINDOWS = 128           # 128 windows x 128 predicted characters per text
EVAL_SEED = 29


def clean(s: str) -> str:
    s = "".join(MAP.get(c, c) for c in s)
    return "".join(c for c in s if c == "\n" or 32 <= ord(c) < 127)


def gsm_text(split: str) -> str:
    rows = parquet(GSM8K, f"main/{split}-00000-of-00001.parquet")
    return "".join(f"Q: {clean(r['question'])}\nA: {clean(r['answer'])}\n\n" for r in rows)


@lru_cache(maxsize=1)
def texts() -> dict[str, str]:
    shakespeare = (CORPUS / "tiny_shakespeare.txt").read_text()
    n_val = int(len(shakespeare) * 0.1)
    return {
        "a_train": shakespeare[:-n_val],
        "a_val": shakespeare[-n_val:],
        "b_train": gsm_text("train"),
        "b_val": gsm_text("test"),
    }


@lru_cache(maxsize=1)
def vocab() -> list[str]:
    t = texts()
    return sorted(set(t["a_train"] + t["a_val"] + t["b_train"] + t["b_val"]))


def encode(s: str) -> torch.Tensor:
    stoi = {c: i for i, c in enumerate(vocab())}
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)


def decode(ids) -> str:
    v = vocab()
    return "".join(v[int(i)] for i in ids)


@lru_cache(maxsize=1)
def tensors() -> dict[str, torch.Tensor]:
    return {k: encode(v) for k, v in texts().items()}


def batch(data: torch.Tensor, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - BLOCK - 1, (n,), generator=gen)
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
    return x, y


@lru_cache(maxsize=1)
def eval_windows() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The fixed held-out windows every run is scored on: 16,384 characters per text."""
    t = tensors()
    out = {}
    for name, key in (("old", "a_val"), ("new", "b_val")):
        g = torch.Generator().manual_seed(EVAL_SEED)
        out[name] = batch(t[key], N_EVAL_WINDOWS, g)
    return out
