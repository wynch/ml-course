"""The first 20,000,000 characters of wikitext-103 (train), as character ids.

Module 19's approach: the train split is two Parquet files of about 157 MB
each, cut into row groups of 1,000 lines. We read row groups from the start of
the first file until the text holds N_CHARS characters (about 12 MB of the
file), and cache the text beside the Hugging Face cache. Module 19 caches a
longer slice of the same file under another name; when it is there we read our
characters from it and need no network.

The vocabulary is every character seen at least MIN_COUNT times in the
training part, plus one id for all rarer ones. The ids are cached as a .npy
file next to the text.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-103-raw-v1/train-00000-of-00002.parquet"

N_CHARS = 20_000_000
N_HELD = 200_000       # the last 200,000 characters: the first half tunes, the second half reports
MIN_COUNT = 10


def _cache_dir() -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "mlcourse"


def load_text(n_chars: int = N_CHARS) -> str:
    rev = WIKITEXT_REVISION[:7]
    own = _cache_dir() / f"wikitext103-train-{rev}-chars-{n_chars}.txt"
    if own.exists():
        return own.read_text()[:n_chars]
    for other in sorted(_cache_dir().glob(f"wikitext103-train-{rev}-*.txt")):
        if "rowgroups" in other.name:
            continue
        text = other.read_text()
        if len(text) >= n_chars:
            return text[:n_chars]

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    url = f"datasets/{WIKITEXT_ID}@{WIKITEXT_REVISION}/{WIKITEXT_FILE}"
    parts, count, group = [], 0, 0
    with HfFileSystem().open(url, "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        while count < n_chars:
            chunk = "".join(pf.read_row_group(group, columns=["text"]).column("text").to_pylist())
            parts.append(chunk)
            count += len(chunk)
            group += 1
    text = "".join(parts)[:n_chars]
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_text(text)
    return text


class CharVocab:
    """Every character seen at least MIN_COUNT times in the training part, sorted, plus UNK."""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.unk = len(chars)
        self.size = len(chars) + 1

    @classmethod
    def from_text(cls, train_text: str, min_count: int = MIN_COUNT) -> "CharVocab":
        counts: dict[str, int] = {}
        for ch in train_text:
            counts[ch] = counts.get(ch, 0) + 1
        return cls(sorted(c for c, k in counts.items() if k >= min_count))

    def encode(self, s: str) -> np.ndarray:
        return np.fromiter((self.stoi.get(c, self.unk) for c in s), dtype=np.int64, count=len(s))

    def decode(self, ids) -> str:
        return "".join(self.chars[i] if i < self.unk else "�" for i in ids)


def splits() -> tuple[np.ndarray, np.ndarray, np.ndarray, CharVocab]:
    """(train, tune, report, vocab): 19.8M training characters, then two held-out halves of 100,000."""
    import json

    rev = WIKITEXT_REVISION[:7]
    ids_path = _cache_dir() / f"wikitext103-train-{rev}-chars-{N_CHARS}-ids.npy"
    voc_path = ids_path.with_suffix(".json")
    if ids_path.exists() and voc_path.exists():
        ids = np.load(ids_path)
        vocab = CharVocab(json.loads(voc_path.read_text()))
    else:
        text = load_text()
        assert len(text) == N_CHARS, len(text)
        vocab = CharVocab.from_text(text[:-N_HELD])
        ids = vocab.encode(text)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(ids_path, ids)
        voc_path.write_text(json.dumps(vocab.chars))
    half = N_HELD // 2
    return ids[:-N_HELD], ids[-N_HELD:-half], ids[-half:], vocab
