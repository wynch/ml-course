"""Section 4, the library way: mlx-lm's own quantised KV cache.

mlx-lm ships `QuantizedKVCache`: it stores keys and values as integers with
`mx.quantize` (per token, groups of 64, a bf16 scale and bias per group) and
attends to them with `mx.quantized_matmul`, without rebuilding the floats.
`generate` turns it on with `kv_bits=`.

Three measures:

1. Perplexity at 8, 4, 3 and 2 bits, the same 32 x 1,024 tokens as section 3,
   beside section 3's hand rule in the same layout (keys and values per
   token, no rotation). Same rule, so the gap is the kernels' arithmetic.
2. Bytes: the cache after a prompt of 1,024, 4,096 and 8,192 tokens of the
   wikitext slice, read from each cache object's `nbytes`.
3. Speed: a 4,096-token prompt, then 64 greedy tokens with the end-of-text
   token blocked; a warm-up, then the median of three, for the bf16 cache and
   the 8- and 4-bit caches.

Run me (after kv_hand.py):
    uv run python src/kv_mlx.py            # numbers, results/kv_mlx.json
    uv run python src/kv_mlx.py --figures  # also figures/kv_mlx.png
"""

from __future__ import annotations

import argparse
import statistics

import common  # noqa: F401
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from common import Guard, ppl_ids, read_results, seed_all, write_results
from kv_hand import load_mlx, perplexity

BITS = (8, 4, 3, 2)
CONTEXTS = (1024, 4096, 8192)
SPEED_PROMPT, SPEED_NEW, SPEED_REPEATS = 4096, 64, 3
PREFILL = 512


def long_ids(n: int) -> list[int]:
    """The first n tokens of the wikitext slice, one sequence."""
    ids = ppl_ids().flatten().tolist()
    assert len(ids) >= n
    return ids[:n]


def cache_bytes(model, ids: list[int], make_cache) -> int:
    caches = [make_cache() for _ in model.layers]
    for i in range(0, len(ids), PREFILL):
        model(mx.array([ids[i : i + PREFILL]]), cache=caches)
        mx.eval([(c.keys, c.values) for c in caches])
    n = sum(c.nbytes for c in caches)
    del caches
    mx.clear_cache()
    return int(n)


def speed(model, tok, prompt: list[int], kv_bits) -> dict:
    from mlx_lm import stream_generate

    vocab = model.model.embed_tokens.weight.shape[0]
    mask = np.zeros(vocab, dtype=np.float32)
    mask[[tok.eos_token_id]] = -np.inf
    mask = mx.array(mask)

    def block_eos(_tokens, logits):
        return logits + mask.astype(logits.dtype)

    def run():
        last = None
        kw = {} if kv_bits is None else {"kv_bits": kv_bits, "kv_group_size": 64, "quantized_kv_start": 0}
        for last in stream_generate(model, tok, prompt, max_tokens=SPEED_NEW, logits_processors=[block_eos], **kw):
            pass
        return last

    run()
    outs = [run() for _ in range(SPEED_REPEATS)]
    assert all(o.generation_tokens == SPEED_NEW for o in outs)
    rates = [o.generation_tps for o in outs]
    return {"median_tok_s": round(statistics.median(rates), 1), "all_tok_s": [round(r, 1) for r in rates],
            "peak_gb": round(max(o.peak_memory for o in outs), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    guard = Guard("kv_mlx.py")
    seed_all()
    ids = ppl_ids()
    model, tok = load_mlx()
    hand = {r["bits"]: r for r in read_results("kv_hand")["rows"]
            if r["bits"] == 16 or (r["keys"] == "token" and not r["rotation"])}
    guard.tick("load")

    rows = [{"bits": 16, "cache": "KVCache (bf16)", **perplexity(model, ids, KVCache), "hand_ppl": hand[16]["ppl"]}]
    for bits in BITS:
        r = perplexity(model, ids, lambda b=bits: QuantizedKVCache(group_size=64, bits=b))
        rows.append({"bits": bits, "cache": f"QuantizedKVCache({bits} bits, group 64)", **r, "hand_ppl": hand[bits]["ppl"]})
        print(f"{bits} bits: library ppl {r['ppl']:.3f}, hand {hand[bits]['ppl']:.3f}")
    guard.tick("ppl")

    sizes = []
    for n in CONTEXTS:
        prompt = long_ids(n)
        row = {"tokens": n, "bf16": cache_bytes(model, prompt, KVCache)}
        for bits in (8, 4, 2):
            row[f"{bits}-bit"] = cache_bytes(model, prompt, lambda b=bits: QuantizedKVCache(group_size=64, bits=b))
        sizes.append(row)
        print(f"cache bytes at {n} tokens: {row}")
    guard.tick("bytes")

    prompt = long_ids(SPEED_PROMPT)
    speeds = []
    for kv_bits in (None, 8, 4):
        s = speed(model, tok, prompt, kv_bits)
        speeds.append({"cache": "bf16" if kv_bits is None else f"{kv_bits}-bit", **s})
        print(f"speed, {speeds[-1]['cache']} cache: {s['median_tok_s']} tok/s")
    guard.tick("speed")

    cfg = model.args
    per_token_bf16 = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2
    out = {
        "rows": rows, "ppl_tokens": rows[0]["tokens"], "cache_bytes": sizes,
        "bf16_bytes_per_token": per_token_bf16,
        "speed_setup": {"prompt_tokens": SPEED_PROMPT, "new_tokens": SPEED_NEW, "repeats": SPEED_REPEATS,
                        "greedy": True, "eos_blocked": True, "kv_group_size": 64, "quantized_kv_start": 0},
        "speed": speeds,
        "mlx": {"mlx": mx.__version__, "mlx_lm": __import__("mlx_lm").__version__},
    }
    g = guard.report()
    g["mlx_peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    out["guard"] = g
    write_results("kv_mlx", out)
    if args.figures:
        import figures

        figures.kv_mlx()


if __name__ == "__main__":
    main()
