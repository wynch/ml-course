"""Reference solution, exercise (d): the window mask, the reach of a stack, and the cache a local/global mix keeps."""

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
    t = torch.arange(T)[:, None]
    s = torch.arange(T)[None, :]
    ok = s <= t
    return ok if w is None else ok & (t - s < w)


def reach(windows: list[int | None]) -> float:
    """How many tokens back the last layer can draw on, through the whole stack."""
    if any(w is None for w in windows):
        return float("inf")
    return sum(w - 1 for w in windows)


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
    n_local, n_global = kinds.count("local"), kinds.count("global")
    kept = n_local * min(context, w) + n_global * context
    return {"local": n_local, "global": n_global, "window": w, "share": kept / (len(kinds) * context)}


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
