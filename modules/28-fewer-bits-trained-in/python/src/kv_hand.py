"""Section 3: SmolLM2-360M-Instruct's KV cache rounded to 8, 4, 3 and 2 bits, by hand.

The model runs in MLX, in bf16, from the cached mlx-community copy (module 23
showed it holds the same weights as the Hugging Face checkpoint). The weights
stay untouched; only the keys and values the model writes into its cache are
rounded (src/kvquant.py). Settings:

    bits         16 (untouched), 8, 4, 3, 2
    keys         per token or per channel (groups of 64 channels, or of 64 tokens)
    values       per token
    rotation     none, or one random 64 x 64 Hadamard matrix shared by every layer and head

Perplexity is module 23's: the first 32 x 1,024 tokens of the wikitext slice,
windows that do not overlap, 32,736 predicted tokens, mean loss in float32.

It also measures the outliers: for every layer, the largest ratio, over the
5 key heads, of the top channel's max |k| to the median channel's, on the
first window, before and after rotation.

Run me:
    uv run python src/kv_hand.py            # numbers, results/kv_hand.json
    uv run python src/kv_hand.py --figures  # also figures/kv_hand.png
"""

from __future__ import annotations

import argparse

import common  # noqa: F401  (guard env; torch runs only on the CPU here)
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from common import MLX_ID, MLX_REVISION, Guard, ppl_ids, seed_all, write_results
from kvquant import FakeQuantCache, RecordingCache, hadamard

BITS = (8, 4, 3, 2)
LAYOUTS = ("channel", "token")
ROT_SEED = 0


def load_mlx():
    from mlx_lm import load

    return load(MLX_ID, revision=MLX_REVISION)


def perplexity(model, ids, make_cache=None, batch: int = 2) -> dict:
    """Mean next-token loss over every window; make_cache() gives one fresh cache per layer, or None."""
    n_layers = len(model.layers)
    total, count = 0.0, 0
    for i in range(0, len(ids), batch):
        x = mx.array(ids[i : i + batch].numpy())
        cache = None if make_cache is None else [make_cache() for _ in range(n_layers)]
        logits = model(x, cache=cache).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits[:, :-1], x[:, 1:], reduction="sum")
        total += float(nll.item())
        count += int(x[:, 1:].size)
        mx.clear_cache()
    loss = total / count
    return {"loss": round(loss, 5), "ppl": round(float(np.exp(loss)), 4), "tokens": count}


def outliers(model, ids, R) -> dict:
    """Per layer: max over key heads of (top channel max|k|) / (median channel max|k|), before and after R."""
    caches = [RecordingCache() for _ in model.layers]
    model(mx.array(ids[:1].numpy()), cache=caches)
    rows = []
    keep = None
    for li, c in enumerate(caches):
        k = c.seen[0][0].astype(mx.float32)                     # (kv_heads, tokens, 64)
        out = {}
        for tag, kk in (("before", k), ("after", k @ R)):
            amax = np.array(mx.abs(kk).max(axis=1))             # (kv_heads, 64)
            ratio = amax.max(1) / np.median(amax, 1)
            out[tag] = round(float(ratio.max()), 2)
            out[f"{tag}_head"] = int(ratio.argmax())
        rows.append({"layer": li, **out})
        if keep is None or out["before"] > keep[0]:
            h = out["before_head"]
            keep = (out["before"], li, h, np.array(k[h, :128]), np.array((k @ R)[h, :128]))
    _, li, h, kb, ka = keep
    sample = {"layer": li, "head": h, "tokens": 128,
              "keys_before": np.round(kb, 2).tolist(), "keys_after": np.round(ka, 2).tolist()}
    return {"per_layer": rows, "sample": sample}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    guard = Guard("kv_hand.py")
    seed_all()
    ids = ppl_ids()
    model, _ = load_mlx()
    R = mx.array(hadamard(64, ROT_SEED))
    guard.tick("load")

    rows = [{"bits": 16, "keys": "-", "values": "-", "rotation": False, **perplexity(model, ids)}]
    print(f"untouched ppl {rows[0]['ppl']:.3f}")
    for bits in BITS:
        for layout in LAYOUTS:
            for rot in (False, True):
                r = perplexity(model, ids, lambda: FakeQuantCache(bits, layout, "token", R if rot else None))
                rows.append({"bits": bits, "keys": layout, "values": "token", "rotation": rot, **r})
                print(f"  {bits} bits, keys per {layout:7s} rotation {str(rot):5s} ppl {r['ppl']:.3f}")
        guard.tick(f"bits{bits}")

    out_stats = outliers(model, ids, R)
    guard.tick("outliers")
    cfg = model.args
    out = {
        "model": MLX_ID, "revision": MLX_REVISION, "dtype": "bfloat16",
        "layers": cfg.num_hidden_layers, "kv_heads": cfg.num_key_value_heads, "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "group": 64, "rotation_seed": ROT_SEED, "ppl_tokens": rows[0]["tokens"],
        "rows": rows, "outliers": out_stats,
        "mlx": {"mlx": mx.__version__, "mlx_lm": __import__("mlx_lm").__version__},
    }
    g = guard.report()
    g["mlx_peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    out["guard"] = g
    write_results("kv_hand", out)
    if args.figures:
        import figures

        figures.kv_hand()


if __name__ == "__main__":
    main()
