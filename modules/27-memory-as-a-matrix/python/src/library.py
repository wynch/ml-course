"""Section 5, the library way: a Gated DeltaNet hybrid against a transformer, on MLX.

Qwen3.5-0.8B mixes three Gated DeltaNet layers with one attention layer, six
times over (24 layers). Qwen3-0.6B is a plain transformer of 28 attention
layers. Both run here with 4-bit weights. For contexts of 512, 4,096 and
16,384 tokens, each model reads a long text with a pass code hidden near the
start, then writes the next 24 tokens greedily. We record:

- the bytes its cache holds after the context (KV cache plus any fixed state),
  counted from the cache's own arrays;
- the time to read the context (prefill) and new tokens per second (decode),
  the median of three trials, since other jobs share the Mac's GPU;
- whether the reply starts with the pass code.

Each model runs as its own chunk, under five minutes, cached in
results/library/<model>.json; a last call merges them:

    uv run python src/library.py qwen3
    uv run python src/library.py qwen3.5
    uv run python src/library.py merge --figures
"""

from __future__ import annotations

import argparse
import json
import time

import common

MODELS = {
    "qwen3": ("mlx-community/Qwen3-0.6B-4bit", "73e3e38d981303bc594367cd910ea6eb48349da8"),
    "qwen3.5": ("mlx-community/Qwen3.5-0.8B-4bit", "da28692b5f139cb0ec58a356b437486b7dac7462"),
}
CONTEXTS = [512, 4096, 16384]
CODES = ["4817", "2093", "7365"]          # one pass code per trial
NEW_TOKENS = 24
PREFILL_STEP = 512
OUT = common.RESULTS / "library"
CORPUS = common.MODULE.parent / "04-attention-transformer" / "corpus" / "tiny_shakespeare.txt"


def cache_bytes(cache) -> dict:
    """Bytes the cache holds for the tokens read so far, split by kind of layer.

    mlx-lm grows a KV cache in blocks of 256 tokens; "kv_used" counts only the
    filled part, "kv_allocated" the whole buffer.
    """
    from mlx_lm.models.cache import ArraysCache, KVCache

    out = {"kv_used": 0, "kv_allocated": 0, "state": 0, "kv_layers": 0, "state_layers": 0}
    for c in cache:
        if isinstance(c, KVCache):
            out["kv_layers"] += 1
            if c.keys is not None:
                k, v = c.keys_and_values()
                out["kv_used"] += k.nbytes + v.nbytes
                out["kv_allocated"] += c.nbytes
        elif isinstance(c, ArraysCache):
            out["state_layers"] += 1
            out["state"] += c.nbytes
        else:
            raise TypeError(type(c))
    out["total_used"] = out["kv_used"] + out["state"]
    return out


def build_prompt(tok, n_tokens: int, code: str, seed: int) -> list[int]:
    """Filler text from tiny-shakespeare, a pass code near the start, a cue at the end.

    Exactly n_tokens tokens. The code sentence sits after the first 5% of the filler.
    """
    text = CORPUS.read_text()
    start = (seed * 100_003) % (len(text) // 2)
    filler = tok.encode(text[start:start + 8 * n_tokens])
    needle = tok.encode(f"\n\nRemember this: the pass code is {code}.\n\n")
    cue = tok.encode("\n\nWhat was the pass code? The pass code is")
    body = n_tokens - len(needle) - len(cue)
    at = body // 20
    ids = filler[:at] + needle + filler[at:body] + cue
    assert len(ids) == n_tokens, (len(ids), n_tokens)
    return ids


def run_model(key: str) -> dict:
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    guard = common.Guard(f"library_{key}")
    repo, rev = MODELS[key]
    path = snapshot_download(repo, revision=rev)
    # MLX keeps freed buffers in a cache for reuse; uncapped, that cache held
    # about 10 GB of Metal memory in a first run. Cap it, and empty it after each trial.
    mx.set_cache_limit(256 * 2**20)
    model, tok = load(path)
    guard.tick("load")
    mx.reset_peak_memory()
    rows = {}
    for L in CONTEXTS:
        trials = []
        for i, code in enumerate(CODES):
            ids = build_prompt(tok, L, code, seed=i)
            cache = make_prompt_cache(model)
            t0 = time.perf_counter()
            for s in range(0, L - 1, PREFILL_STEP):                    # all but the last token
                chunk = mx.array(ids[s:min(s + PREFILL_STEP, L - 1)])[None]
                mx.eval(model(chunk, cache=cache))
            prefill_s = time.perf_counter() - t0
            y = mx.array(ids[-1:])[None]
            out_ids, t1 = [], None
            for j in range(NEW_TOKENS):
                logits = model(y, cache=cache)
                y = mx.argmax(logits[:, -1, :], axis=-1)[None]
                mx.eval(y)
                out_ids.append(int(y.item()))
                if j == 0:
                    mem = cache_bytes(cache)                            # the cache now holds all L tokens
                    t1 = time.perf_counter()                            # time from the 2nd token on
            decode_tps = (NEW_TOKENS - 1) / (time.perf_counter() - t1)
            reply = tok.decode(out_ids)
            trials.append({"code": code, "reply": reply, "found": reply.strip().startswith(code),
                           "prefill_s": round(prefill_s, 3), "decode_tokens_per_s": round(decode_tps, 1),
                           "cache": mem})
            print(f"{key} L={L} code={code} prefill {prefill_s:.2f}s decode {decode_tps:.1f} tok/s "
                  f"cache {mem['total_used']:,} B reply {reply[:30]!r}", flush=True)
            del cache, logits, y
            mx.clear_cache()
            guard.tick()
        rows[str(L)] = {
            "trials": trials,
            "found": sum(t["found"] for t in trials),
            "cache_bytes": trials[0]["cache"]["total_used"],
            "prefill_s_median": sorted(t["prefill_s"] for t in trials)[1],
            "decode_tokens_per_s_median": sorted(t["decode_tokens_per_s"] for t in trials)[1],
        }
        guard.tick(f"L{L}")
    cfg = json.loads(open(f"{path}/config.json").read())
    tc = cfg.get("text_config", cfg)
    payload = {
        "model": repo, "revision": rev, "config": {k: tc.get(k) for k in (
            "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
            "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
            "linear_conv_kernel_dim", "full_attention_interval", "vocab_size")},
        "layer_types": tc.get("layer_types"),
        "contexts": rows,
        "guard": guard.report(peak_mlx_gb=round(mx.get_peak_memory() / 2**30, 2), mlx=mx.__version__),
    }
    common.write_results(key, payload, OUT)
    return payload


def merge(figures: bool) -> None:
    parts = {k: json.loads((OUT / f"{k}.json").read_text()) for k in MODELS}
    out = {"contexts": CONTEXTS, "new_tokens": NEW_TOKENS, "prefill_step": PREFILL_STEP, "models": parts}
    for k, p in parts.items():
        c = p["contexts"]
        a, b = c[str(CONTEXTS[0])]["cache_bytes"], c[str(CONTEXTS[-1])]["cache_bytes"]
        out.setdefault("bytes_per_token", {})[k] = round((b - a) / (CONTEXTS[-1] - CONTEXTS[0]), 1)
        out.setdefault("fixed_bytes", {})[k] = round(a - out["bytes_per_token"][k] * CONTEXTS[0])
    common.write_results("library", out)
    if figures:
        import figures as F

        F.fig_library()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=[*MODELS, "merge"])
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.what == "merge":
        merge(args.figures)
    else:
        run_model(args.what)


if __name__ == "__main__":
    main()
