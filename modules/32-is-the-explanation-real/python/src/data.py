"""TinyStories, tokenised with GPT-2's tokenizer and cut to a 4,096-token vocabulary.

Source: ``roneneldan/TinyStories``, file ``TinyStories-valid.txt`` (19.4 MB,
21,990 stories split by ``<|endoftext|>``), pinned revision below. We use that
one file for everything: the first 90% of its stories train, the last 10% are
held out. GPT-2's tokenizer splits it into about 4.7M tokens; we keep the
4,095 most frequent token ids of the training stories, plus id 0 for any other
token (``<unk>``). The story separator stays a token, so the model sees where
stories end.

The token arrays are cached in ``artifacts/runs/tokens.npz`` (not committed).
"""

from __future__ import annotations

import json

import numpy as np

from common import ARTIFACTS, RUNS

REPO = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
FILE = "TinyStories-valid.txt"
VOCAB = 4096
SEP = "<|endoftext|>"
VOCAB_FILE = ARTIFACTS / "vocab.json"                # committed: our id -> GPT-2 id


def raw_text() -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    return open(path, encoding="utf-8").read()


def stories() -> list[str]:
    parts = [s.strip() for s in raw_text().split(SEP)]
    return [s for s in parts if s]


def gpt2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


def build() -> dict:
    """Tokenise, build the vocabulary from the training stories, cache both."""
    tok = gpt2_tokenizer()
    st = stories()
    n_train = int(len(st) * 0.9)
    sep_id = tok.eos_token_id

    def ids(chunk):
        out = []
        for s in chunk:
            out.extend(tok.encode(s))
            out.append(sep_id)
        return np.array(out, dtype=np.int64)

    tr, va = ids(st[:n_train]), ids(st[n_train:])
    counts = np.bincount(tr, minlength=tok.vocab_size)
    keep = np.argsort(-counts, kind="stable")[: VOCAB - 1]
    gpt2_of = np.concatenate([[-1], keep])               # our id 0 is <unk>
    ours = np.zeros(tok.vocab_size, dtype=np.int64)
    ours[keep] = np.arange(1, VOCAB)
    meta = {
        "source": f"{REPO}@{REVISION}/{FILE}",
        "stories": len(st), "train_stories": n_train, "val_stories": len(st) - n_train,
        "train_tokens": int(len(tr)), "val_tokens": int(len(va)),
        "unk_share_train": round(float((ours[tr] == 0).mean()), 5),
        "unk_share_val": round(float((ours[va] == 0).mean()), 5),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    VOCAB_FILE.write_text(json.dumps({"gpt2_ids": gpt2_of.tolist(), **meta}) + "\n")
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez(RUNS / "tokens.npz", train=ours[tr].astype(np.int16), val=ours[va].astype(np.int16))
    return meta


def tokens() -> tuple[np.ndarray, np.ndarray]:
    path = RUNS / "tokens.npz"
    if not path.exists():
        build()
    z = np.load(path)
    return z["train"].astype(np.int64), z["val"].astype(np.int64)


class Vocab:
    """Our ids to readable strings, through GPT-2's tokenizer."""

    def __init__(self):
        self.gpt2_ids = json.loads(VOCAB_FILE.read_text())["gpt2_ids"]
        self._tok = None

    def piece(self, i: int) -> str:
        if i == 0:
            return "<unk>"
        if self._tok is None:
            self._tok = gpt2_tokenizer()
        return self._tok.decode([self.gpt2_ids[i]])

    def pieces(self) -> list[str]:
        if self._tok is None:
            self._tok = gpt2_tokenizer()
        return ["<unk>"] + [self._tok.decode([g]) for g in self.gpt2_ids[1:]]

    def encode(self, text: str) -> list[int]:
        if self._tok is None:
            self._tok = gpt2_tokenizer()
        lut = {g: i for i, g in enumerate(self.gpt2_ids) if i}
        return [lut.get(g, 0) for g in self._tok.encode(text)]
