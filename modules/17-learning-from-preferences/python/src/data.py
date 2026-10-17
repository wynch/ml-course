"""Preference pairs from trl-lib/ultrafeedback_binarized, fetched by the bytes they need.

The train split is one 131 MB Parquet file cut into row groups of 1,000 pairs
(about 2 MB each); the test split is one 2.1 MB file holding a single row
group of 1,000 pairs. We open both over HTTP and read only the row groups we
need, then cache the pairs as JSON lines beside the Hugging Face cache, so
later runs need no network.

Each pair is one prompt, a chosen reply and a rejected reply, with the GPT-4
ratings UltraFeedback gave them. We keep single-turn pairs whose prompt and
both replies fit in MAX_TOKENS once put in SmolLM2's chat template, whose two
replies differ, and whose two ratings differ (on a tie the label is arbitrary).

    uv run python src/data.py      # fetch, filter, print the counts
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DATASET_ID = "trl-lib/ultrafeedback_binarized"
DATASET_REVISION = "47124cb5778f5d50de1c7676a412828f3ea7c555"
TRAIN_FILE = "data/train-00000-of-00001.parquet"
TEST_FILE = "data/test-00000-of-00001.parquet"
TRAIN_GROUPS = 8            # 8,000 train pairs
MAX_TOKENS = 640            # prompt + reply, in SmolLM2's chat template

N_REWARD_TRAIN = 2000       # section 1: the reward head trains on these
N_HELDOUT = 500             # sections 1, 3, 4: held-out pairs from the test split


def cache_dir() -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "mlcourse" / f"ultrafeedback-binarized-{DATASET_REVISION[:7]}"


def _read_groups(path_in_repo: str, groups: range) -> tuple[list[dict], int]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    url = f"datasets/{DATASET_ID}@{DATASET_REVISION}/{path_in_repo}"
    rows: list[dict] = []
    with HfFileSystem().open(url, "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        for g in groups:
            rows += pf.read_row_group(g).to_pylist()
        read_to = f.loc
    return rows, read_to


def _to_pair(row: dict, i: int, split: str) -> dict | None:
    c, r = row["chosen"], row["rejected"]
    if len(c) != 2 or len(r) != 2 or c[0]["role"] != "user" or c[1]["role"] != "assistant":
        return None                                    # keep single-turn pairs only
    if c[0]["content"] != r[0]["content"]:
        return None
    return {"id": f"{split}-{i}", "prompt": c[0]["content"], "chosen": c[1]["content"],
            "rejected": r[1]["content"], "score_chosen": row["score_chosen"],
            "score_rejected": row["score_rejected"]}


def fetch_raw() -> tuple[dict[str, list[dict]], dict]:
    """All pairs in the row groups read, before any filter."""
    d = cache_dir()
    meta = d / "fetch.json"
    if meta.exists():
        out = {s: [json.loads(l) for l in (d / f"{s}.jsonl").read_text().split("\n") if l] for s in ("train", "test")}
        return out, json.loads(meta.read_text())
    train_rows, train_bytes = _read_groups(TRAIN_FILE, range(TRAIN_GROUPS))
    test_rows, test_bytes = _read_groups(TEST_FILE, range(1))
    out = {"train": [p for i, r in enumerate(train_rows) if (p := _to_pair(r, i, "train"))],
           "test": [p for i, r in enumerate(test_rows) if (p := _to_pair(r, i, "test"))]}
    info = {"dataset": DATASET_ID, "revision": DATASET_REVISION,
            "train_row_groups": TRAIN_GROUPS, "test_row_groups": 1,
            "train_rows": len(train_rows), "test_rows": len(test_rows),
            "bytes_read": train_bytes + test_bytes}
    d.mkdir(parents=True, exist_ok=True)
    for s in out:
        (d / f"{s}.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in out[s]))
    meta.write_text(json.dumps(info))
    return out, info


def n_tokens(tok, prompt: str, reply: str) -> int:
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
    return len(tok.apply_chat_template(msgs, tokenize=True, return_dict=False))


def load_pairs(tok=None) -> tuple[dict[str, list[dict]], dict]:
    """Filtered pairs: {'train': [...], 'test': [...]}, plus counts.

    Adds the token count of each reply in its chat template (``len_chosen``,
    ``len_rejected``, prompt included) and the reply's own length in
    characters and words.
    """
    d = cache_dir()
    path = d / f"filtered-{MAX_TOKENS}.json"
    if path.exists():
        blob = json.loads(path.read_text())
        return blob["pairs"], blob["info"]
    if tok is None:
        from common import load_tokenizer
        tok = load_tokenizer()
    raw, info = fetch_raw()
    pairs, counts = {}, {}
    for s, rows in raw.items():
        keep = []
        for p in rows:
            if p["chosen"].strip() == p["rejected"].strip():
                continue
            if p["score_chosen"] == p["score_rejected"]:
                continue                               # a tie: the chosen label is arbitrary
            lc, lr = n_tokens(tok, p["prompt"], p["chosen"]), n_tokens(tok, p["prompt"], p["rejected"])
            if max(lc, lr) > MAX_TOKENS:
                continue
            p = dict(p, len_chosen=lc, len_rejected=lr,
                     reply_tokens_chosen=len(tok(p["chosen"], add_special_tokens=False)["input_ids"]),
                     reply_tokens_rejected=len(tok(p["rejected"], add_special_tokens=False)["input_ids"]))
            keep.append(p)
        pairs[s] = keep
        counts[s] = {"single_turn": len(rows), "kept": len(keep)}
    info = dict(info, max_tokens=MAX_TOKENS, counts=counts)
    path.write_text(json.dumps({"pairs": pairs, "info": info}, ensure_ascii=False))
    return pairs, info


def splits(pairs: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """reward_train: first 2,000 kept train pairs; dpo_train: the rest; heldout: first 500 kept test pairs."""
    return {"reward_train": pairs["train"][:N_REWARD_TRAIN],
            "dpo_train": pairs["train"][N_REWARD_TRAIN:],
            "heldout": pairs["test"][:N_HELDOUT]}


if __name__ == "__main__":
    raw, info = fetch_raw()
    print(f"read {info['bytes_read'] / 2**20:.1f} MB: {info['train_rows']} train rows, {info['test_rows']} test rows")
    pairs, info = load_pairs()
    for s, c in info["counts"].items():
        print(f"  {s}: {c['single_turn']} single-turn, {c['kept']} kept (<= {MAX_TOKENS} tokens)")
    print({k: len(v) for k, v in splits(pairs).items()})
