"""Exercise (d): the window mask, the reach of a stack, and the cache a local/global mix keeps.

A window of w lets query t read keys t - w + 1 to t. Build that mask. Then
work out how far back the last layer of a stack can draw on: each window
layer adds w - 1 tokens, and one full layer makes it unlimited. Last, read
Gemma 3 27B's and Inkling 2 Small's records from the architecture explorer's
database and count the KV cache their local/global mix keeps.

Predict before you run: Gemma 3 has five local layers (window 1,024) for
each global one. At a context of 131,072 tokens, what share of an
all-global cache does it keep: about 5/6, 1/2, or under 1/5?

    uv run python ../exercises/exercise_d_window.py
    uv run pytest tests/test_solutions.py -k window -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch

RECORDS = Path(__file__).resolve().parents[3] / "architectures" / "records"


def window_mask(T: int, w: int | None) -> torch.Tensor:
    """(T, T) bool: query t may read key s when s <= t and t - s < w. w None: plain causal."""
    # TODO(you): two aranges, one as a column and one as a row; causal, and inside the window.
    raise NotImplementedError("TODO(you): write window_mask first")


def reach(windows: list[int | None]) -> float:
    """How many tokens back the last layer can draw on, through the whole stack."""
    # TODO(you): inf if any layer is full (None); else the sum of w - 1.
    raise NotImplementedError("TODO(you): write reach first")


def layer_kinds(record: dict) -> list[str]:
    """The record's stack, one entry per layer: prefix, pattern x repeat, suffix."""
    st = record["stack"]
    out = []
    for part in st.get("prefix", []) + st["pattern"] * st["repeat"] + st.get("suffix", []):
        out += [part["type"]] * part["count"]
    return out


def kv_share(record: dict, context: int) -> dict:
    """At a context of n tokens, the KV cache a local/global mix keeps, against all layers full.
    A local layer keeps min(n, window) tokens; a global layer keeps n."""
    kinds = layer_kinds(record)
    w = record["attention"]["window"]
    # TODO(you): count "local" and "global" in kinds, then the tokens kept against len(kinds) * context.
    raise NotImplementedError("TODO(you): write kv_share first")


def main() -> None:
    m = window_mask(8, 3).int()
    print("window 3 over 8 tokens:\n", m)
    for ws in ([8, 8, 8], [16, 16, 16], [32, 32, 32], [8, 8, None]):
        print(f"windows {ws}: reach {reach(ws)}")
    for rid in ("gemma-3-27b", "inkling-2-small"):
        r = json.loads((RECORDS / f"{rid}.json").read_text())
        n = 131072
        s = kv_share(r, n)
        print(f"{r['name']}: {s['local']} local (w {s['window']}) + {s['global']} global; "
              f"at {n:,} tokens it keeps {s['share']:.1%} of an all-global cache")


if __name__ == "__main__":
    main()
