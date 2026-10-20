"""The first row groups of wikitext-103 (train), fetched by the bytes they need.

Module 11's approach: the train split is two Parquet files of about 157 MB
each, cut into row groups of 1,000 lines. We open the first file over HTTP and
read only the first ``n_groups`` row groups. The text is cached as plain text
beside the Hugging Face cache, so later runs need no network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
N_GROUPS = 2


def cache_path(n_groups: int = N_GROUPS) -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "mlcourse" / f"wikitext103-train-{WIKITEXT_REVISION[:7]}-rowgroups-{n_groups}.txt"


def fetch_text(n_groups: int = N_GROUPS) -> tuple[str, dict]:
    """Return (text, info) for the first ``n_groups`` row groups."""
    path = cache_path(n_groups)
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        return path.read_text(), json.loads(meta.read_text())

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    url = f"datasets/{WIKITEXT_ID}@{WIKITEXT_REVISION}/{WIKITEXT_FILE}"
    with HfFileSystem().open(url, "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        lines = []
        for g in range(n_groups):
            lines += pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
        read_to = f.loc
    text = "".join(lines)
    info = {"row_groups_read": n_groups, "lines": len(lines), "chars": len(text), "read_to_byte": read_to}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    meta.write_text(json.dumps(info))
    return text, info
