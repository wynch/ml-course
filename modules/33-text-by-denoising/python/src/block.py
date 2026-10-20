"""Section 4: block diffusion with a KV cache.

Four models, one network size, one recipe; only the training loss differs:
ar (L = 1 in effect), bd4, bd16, mdm (L = 128). Each writes 64 lines with
one character per step (128 steps) and with four characters per step (32
steps), most-confident-first, temperature 1, scored as in section 3. The
block models run twice, with and without the KV cache, from the same random
draws: the characters should match, and the cache should save time. Wall
times are on the GPU (MPS), median of 3 runs, with other jobs on the machine.

    uv run python src/block.py           # writes results/block.json
"""

from __future__ import annotations

import statistics
import time

import common
import torch

import data as D
import diffusion as Df
from sample import scores, sync
from train import load_model

N = 64
REPEATS = 3


def timed(fn):
    """Median wall time of REPEATS calls; returns (last output, seconds)."""
    secs = []
    for _ in range(REPEATS):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        secs.append(time.perf_counter() - t0)
    return out, round(statistics.median(secs), 3)


def main() -> None:
    dev = common.device()
    guard = common.Guard("block", dev)
    tok, train, val = D.load()
    vocab = D.word_set(train, tok)
    lu = D.log_unigram(train, tok.vocab_size)
    m = {a: load_model(a, dev) for a in ("ar", "bd4", "bd16", "mdm")}
    rows = {}
    for total in (128, 32):
        r = {}
        for name, L in (("bd4", 4), ("bd16", 16)):
            spb = total // (128 // L)
            for cache in (True, False):
                ids, secs = timed(lambda: Df.sample_bd(m[name], N, L, spb, gen=torch.Generator().manual_seed(7),
                                                       cache=cache))
                key = f"{name}_{'cache' if cache else 'nocache'}"
                r[key] = {**scores(ids, tok, vocab, m["ar"], lu), "seconds_for_64": secs,
                          **Df.network_calls("bd", total, L, cache=cache)}
                r[key]["ids"] = ids.cpu()
            same = (r[f"{name}_cache"].pop("ids") == r[f"{name}_nocache"].pop("ids")).float().mean()
            r[f"{name}_cache"]["same_chars_as_nocache"] = round(float(same), 5)
        ids, secs = timed(lambda: Df.sample_mdm(m["mdm"], N, total, "confidence", 1.0, gen=torch.Generator().manual_seed(7)))
        r["mdm"] = {**scores(ids, tok, vocab, m["ar"], lu), "seconds_for_64": secs, **Df.network_calls("mdm", total)}
        if total == 128:
            for cache in (True, False):
                ids, secs = timed(lambda: Df.sample_ar(m["ar"], N, lu, gen=torch.Generator().manual_seed(7), cache=cache))
                r[f"ar_{'cache' if cache else 'nocache'}"] = {**scores(ids, tok, vocab, m["ar"], lu), "seconds_for_64": secs,
                                                              **Df.network_calls("ar", total, cache=cache)}
        rows[str(total)] = r
        guard.tick(f"steps_{total}")
        print(total, {k: (v["word_score"], v["seconds_for_64"]) for k, v in r.items()})
    traces = []
    for name, L in (("bd4", 4), ("bd16", 16)):
        for seed in range(4):
            x, when = Df.sample_bd(m[name], 1, L, L, gen=torch.Generator().manual_seed(500 + seed), trace=True)
            traces.append({"model": name, "L": L, "seed": seed, "text": tok.decode(x[0].tolist()), "when": when[0].tolist()})
    guard.tick("traces")
    common.write_results("block", {"n_lines": N, "rule": "confidence", "temperature": 1.0, "repeats": REPEATS,
                                   "results": rows, "traces": traces, "guard": guard.report()})


if __name__ == "__main__":
    main()
