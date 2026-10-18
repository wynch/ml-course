"""The first 5,000,000 characters of wikitext-103 (train), and a character vocabulary.

The train split is two Parquet files of about 157 MB each, cut into row groups
of 1,000 lines. We open the first file over HTTP and read row groups until the
text holds 5M characters (about 12 MB of the file), then cache the text beside
the Hugging Face cache. Modules 11 and 12 cache a longer slice of the same
file, from its start, under another name; when that file is there we read our
5M characters from it and need no network.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from common import MIN_COUNT, N_CHARS, N_HELD

WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-103-raw-v1/train-00000-of-00002.parquet"


def _cache_dir() -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "mlcourse"


def load_text(n_chars: int = N_CHARS) -> str:
    rev = WIKITEXT_REVISION[:7]
    own = _cache_dir() / f"wikitext103-train-{rev}-chars-{n_chars}.txt"
    if own.exists():
        return own.read_text()[:n_chars]
    # Modules 11 and 12's slices start at the same first row group.
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

    def __init__(self, train_text: str, min_count: int = MIN_COUNT):
        counts: dict[str, int] = {}
        for ch in train_text:
            counts[ch] = counts.get(ch, 0) + 1
        self.chars = sorted(c for c, k in counts.items() if k >= min_count)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.unk = len(self.chars)
        self.size = len(self.chars) + 1

    def encode(self, s: str) -> np.ndarray:
        return np.fromiter((self.stoi.get(c, self.unk) for c in s), dtype=np.int64, count=len(s))

    def decode(self, ids) -> str:
        return "".join(self.chars[i] if i < self.unk else "�" for i in ids)


def splits() -> tuple[np.ndarray, np.ndarray, CharVocab]:
    """(train ids, held-out ids, vocab): the first 4,950,000 characters and the last 50,000."""
    text = load_text()
    assert len(text) == N_CHARS, len(text)
    train_txt, held_txt = text[:-N_HELD], text[-N_HELD:]
    vocab = CharVocab(train_txt)
    return vocab.encode(train_txt), vocab.encode(held_txt), vocab
