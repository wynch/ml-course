"""A 1M-token slice of wikitext-103 (train), fetched by the byte range it needs.

The train split is two Parquet files of about 157 MB each. A Parquet file is
cut into row groups of 1,000 lines, so we open the first file over HTTP and
read row groups one by one until the text holds one million SmolLM2 tokens.
That reads about 2.6 MB of the file instead of 157 MB. The slice is cached as
plain text beside the Hugging Face cache, so later runs need no network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
N_TOKENS = 1_000_000


def cache_path(n_tokens: int = N_TOKENS) -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "mlcourse" / f"wikitext103-train-{WIKITEXT_REVISION[:7]}-{n_tokens}.txt"


def fetch_text(tokenizer, n_tokens: int = N_TOKENS) -> tuple[str, dict]:
    """Return (text, info); the text holds at least ``n_tokens`` tokens."""
    path = cache_path(n_tokens)
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        return path.read_text(), json.loads(meta.read_text())

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    url = f"datasets/{WIKITEXT_ID}@{WIKITEXT_REVISION}/{WIKITEXT_FILE}"
    parts: list[str] = []
    count = 0
    groups = 0
    with fs.open(url, "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        while count < n_tokens:
            lines = pf.read_row_group(groups, columns=["text"]).column("text").to_pylist()
            chunk = "".join(lines)
            parts.append(chunk)
            count += len(tokenizer(chunk, verbose=False).input_ids)
            groups += 1
        read_to = f.loc                          # how far into the 157 MB file we read
    text = "".join(parts)
    info = {"row_groups_read": groups, "lines": sum(p.count("\n") for p in parts),
            "chars": len(text), "read_to_byte": read_to}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    meta.write_text(json.dumps(info))
    return text, info


def token_chunks(tokenizer, text: str, n_tokens: int = N_TOKENS, seq_len: int = 256):
    """The first ``n_tokens`` tokens, cut into rows of ``seq_len``: (n_seq, seq_len)."""
    import torch

    ids = tokenizer(text, verbose=False).input_ids[:n_tokens]
    n_seq = len(ids) // seq_len
    return torch.tensor(ids[: n_seq * seq_len]).view(n_seq, seq_len)
