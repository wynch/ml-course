"""Section 3: write a 128-character line from all [MASK] in 1 to 128 steps.

For each diffusion model (mdm, and mdm6x at six times the training), each
number of steps and each rule (random: reveal k random masked positions;
confidence: reveal the k the model is surest of), 64 lines at temperature 1,
scored three ways:

  word score   share of complete words found in the training split
  judge NLL    the AR twin's NLL per character of the line (lower = likelier
               to the AR model; a line that repeats itself also scores low)
  distinct 4   distinct 4-grams over all 4-grams in the 64 lines (lower = more repetition)

References: 64 lines from the AR twin (temperature 1) and 64 held-out blocks.
It also writes the unmasking traces the explorable replays.

    uv run python src/sample.py          # writes results/sample.json and results/traces.json
"""

from __future__ import annotations

import time

import common
import torch

import data as D
import diffusion as Df
from train import load_model

STEPS = (1, 2, 4, 8, 16, 32, 64, 128)
RULES = ("random", "confidence")
N = 64
MODELS = ("mdm", "mdm6x")     # the AR twin's compute, and six times it


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def distinct4(lines: list[str]) -> float:
    grams = [ln[i: i + 4] for ln in lines for i in range(len(ln) - 3)]
    return len(set(grams)) / len(grams)


def scores(ids: torch.Tensor, tok, vocab, judge, lu) -> dict:
    lines = [tok.decode(r.tolist()) for r in ids.cpu()]
    return {"word_score": round(sum(D.word_score(t, vocab) for t in lines) / len(lines), 4),
            "judge_nll": round(Df.ar_nll(judge, ids.cpu(), lu), 4),
            "distinct4": round(distinct4(lines), 4)}


def main() -> None:
    dev = common.device()
    guard = common.Guard("sample", dev)
    tok, train, val = D.load()
    vocab = D.word_set(train, tok)
    lu = D.log_unigram(train, tok.vocab_size)
    ar = load_model("ar", dev)
    rows, examples = {}, {}
    for name in MODELS:
        mdm = load_model(name, dev)
        rows[name] = {}
        for rule in RULES:
            rows[name][rule] = {}
            for n in STEPS:
                gen = torch.Generator().manual_seed(100 + n)
                sync()
                t0 = time.perf_counter()
                ids = Df.sample_mdm(mdm, N, n, rule, 1.0, gen=gen)
                sync()
                secs = time.perf_counter() - t0
                rows[name][rule][str(n)] = {**scores(ids, tok, vocab, ar, lu), "seconds_for_64": round(secs, 3)}
                examples[f"{name}_{rule}_{n}"] = [tok.decode(r.tolist()) for r in ids[:3].cpu()]
                print(name, rule, n, rows[name][rule][str(n)])
        guard.tick(name)
    gen = torch.Generator().manual_seed(99)
    sync()
    t0 = time.perf_counter()
    ids = Df.sample_ar(ar, N, lu, gen=gen)
    sync()
    ref = {"ar_samples": {**scores(ids, tok, vocab, ar, lu), "seconds_for_64": round(time.perf_counter() - t0, 3)},
           "held_out_text": scores(D.val_blocks(val, N), tok, vocab, ar, lu)}
    examples["ar"] = [tok.decode(r.tolist()) for r in ids[:3].cpu()]
    examples["held_out"] = [tok.decode(r.tolist()) for r in D.val_blocks(val, N)[:3]]
    guard.tick("references")
    traces = []
    for name in MODELS:
        mdm = load_model(name, dev)
        for rule in RULES:
            for n in STEPS:
                for seed in range(4):
                    x, when = Df.sample_mdm(mdm, 1, n, rule, 1.0, gen=torch.Generator().manual_seed(500 + seed), trace=True)
                    traces.append({"model": name, "rule": rule, "steps": n, "seed": seed, "text": tok.decode(x[0].tolist()),
                                   "when": when[0].tolist()})
    guard.tick("traces")
    common.write_results("traces", {"what": "one line per rule, step count and seed; when[i] = step that revealed character i",
                                    "traces": traces})
    common.write_results("sample", {"n_lines": N, "temperature": 1.0, "steps": list(STEPS), "results": rows,
                                    "references": ref, "examples": examples, "guard": guard.report()})


if __name__ == "__main__":
    main()
