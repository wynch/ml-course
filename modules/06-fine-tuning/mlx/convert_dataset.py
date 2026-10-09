"""Convert smoltalk/everyday-conversations to the jsonl format mlx_lm.lora wants.

mlx_lm.lora reads a directory containing train.jsonl / valid.jsonl (and
optionally test.jsonl). It understands the "chat" schema natively — one JSON
object per line with a "messages" list — which is exactly what smoltalk already
uses, so the conversion is mostly a reshape + a train/valid split.

Run:  uv run python convert_dataset.py --n 512
Produces:  data/train.jsonl  data/valid.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="total examples to keep")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    args = ap.parse_args()

    ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations",
                      split="train")
    ds = ds.shuffle(seed=0).select(range(min(args.n, len(ds))))

    n_valid = max(1, int(len(ds) * args.valid_frac))
    DATA_DIR.mkdir(exist_ok=True)

    def dump(rows, path):
        with open(path, "w") as f:
            for r in rows:
                # keep only role/content — mlx_lm's chat loader expects exactly that
                msgs = [{"role": m["role"], "content": m["content"]}
                        for m in r["messages"]]
                f.write(json.dumps({"messages": msgs}) + "\n")

    valid = [ds[i] for i in range(n_valid)]
    train = [ds[i] for i in range(n_valid, len(ds))]
    dump(train, DATA_DIR / "train.jsonl")
    dump(valid, DATA_DIR / "valid.jsonl")
    print(f"wrote {len(train)} train + {len(valid)} valid rows to {DATA_DIR}")


if __name__ == "__main__":
    main()
