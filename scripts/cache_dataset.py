#!/usr/bin/env python3
"""Materialize one pinned Hugging Face dataset into the selected offline cache."""

from __future__ import annotations

import argparse

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    kwargs = {"revision": args.revision}
    if args.split:
        kwargs["split"] = args.split
    result = load_dataset(args.dataset, args.config or None, **kwargs)
    if hasattr(result, "num_rows"):
        print(f"cached {args.dataset}: {result.num_rows} rows")
    else:
        counts = {name: split.num_rows for name, split in result.items()}
        print(f"cached {args.dataset}: {counts}")


if __name__ == "__main__":
    main()
