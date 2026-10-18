"""Every number of sections 1-3, from the cached generations alone.

    uv run python src/analyse.py [--figures]

Reads results/generations/*.jsonl (written by generate.py) and writes
results/analysis.json. No model is loaded; it runs in a few seconds.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re

from common import GEN, MODEL_ID, MODEL_REVISION, DATASET_ID, DATASET_REVISION, Guard, read_jsonl, write_results
from gsm8k import ANS_RE, NUM_RE, canon, extract, same, vote

N_SAMPLES = 8


def sign_test(b: int, c: int) -> float:
    """Exact two-sided sign test on the discordant counts b and c."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def best_of(answers: list[str | None], scores: list[float]) -> str | None:
    """The answer of the highest-scored sample (scores: log-odds); ties go to the earliest."""
    k = max(range(len(answers)), key=lambda k: (scores[k], -k))
    return answers[k]


def weighted_vote(answers: list[str | None], scores: list[float]) -> str | None:
    """Each sample votes with its verifier score; ties go to the answer seen first."""
    w: dict[str, float] = {}
    first: dict[str, int] = {}
    for i, (a, s) in enumerate(zip(answers, scores)):
        if a is None:
            continue
        w[a] = w.get(a, 0.0) + s
        first.setdefault(a, i)
    if not w:
        return None
    return max(w, key=lambda a: (w[a], -first[a]))


ANSWER_LINE_RE = re.compile(r"answer\W", re.IGNORECASE)


def answer_line(text: str) -> str | None:
    """Diagnostic only, chosen after reading replies: the first number after the last
    word "answer". Never used for a score; it counts where the fixed rule may misread."""
    hits = list(ANSWER_LINE_RE.finditer(text))
    if not hits:
        return None
    nums = NUM_RE.findall(text[hits[-1].end():])
    return canon(nums[0]) if nums else None


def load() -> dict:
    probs = read_jsonl(GEN / "problems.jsonl")
    caches = {name: {r["id"]: r for r in read_jsonl(GEN / f"{name}.jsonl")}
              for name in ("direct", "greedy", "samples", "verify")}
    ids = [p["id"] for p in probs]
    for name, d in caches.items():
        assert all(i in d for i in ids), f"{name}.jsonl is incomplete: {len(d)}/{len(ids)}; run generate.py"
    return {"problems": probs, "ids": ids, **caches}


def analyse(c: dict) -> dict:
    ids = c["ids"]
    N = len(ids)
    gold = {p["id"]: p["gold"] for p in c["problems"]}
    direct = {i: extract(c["direct"][i]["text"]) for i in ids}
    greedy = {i: extract(c["greedy"][i]["text"]) for i in ids}
    samp = {i: [extract(s["text"]) for s in c["samples"][i]["samples"]] for i in ids}
    score = {i: [v["score"] for v in c["verify"][i]["verdicts"]] for i in ids}
    # best-of-n ranks by log-odds, log P(Yes) - log P(No): the same order as the score, but
    # the stored score rounds to 1.0 for most samples and would tie.
    logodds = {i: [round(v["lp_yes"] - v["lp_no"], 5) for v in c["verify"][i]["verdicts"]] for i in ids}
    stok = {i: [s["tokens"] for s in c["samples"][i]["samples"]] for i in ids}
    right = lambda pred: [i for i in ids if same(pred[i], gold[i])]  # noqa: E731
    acc = lambda pred: len(right(pred)) / N  # noqa: E731

    def paired(a: dict, b: dict) -> dict:
        only_a = sum(same(a[i], gold[i]) and not same(b[i], gold[i]) for i in ids)
        only_b = sum(same(b[i], gold[i]) and not same(a[i], gold[i]) for i in ids)
        return {"only_first_right": only_a, "only_second_right": only_b, "sign_test_p": sign_test(only_a, only_b)}

    # Section 1: answer only against chain of thought.
    sec1 = {
        "direct_acc": acc(direct), "greedy_acc": acc(greedy),
        "direct_mean_tokens": sum(c["direct"][i]["tokens"] for i in ids) / N,
        "greedy_mean_tokens": sum(c["greedy"][i]["tokens"] for i in ids) / N,
        "greedy_hit_256": sum(c["greedy"][i]["finish"] == "length" for i in ids),
        "greedy_no_number": sum(greedy[i] is None for i in ids),
        "greedy_with_answer_is": sum(bool(ANS_RE.search(c["greedy"][i]["text"])) for i in ids),
        "direct_vs_greedy": paired(direct, greedy),
        "direct_hit_16": sum(c["direct"][i]["finish"] == "length" for i in ids),
        "greedy_finished": {"problems": sum(c["greedy"][i]["finish"] == "stop" for i in ids),
                            "right": sum(c["greedy"][i]["finish"] == "stop" and same(greedy[i], gold[i]) for i in ids)},
        "greedy_cut_at_256": {"problems": sum(c["greedy"][i]["finish"] == "length" for i in ids),
                              "right": sum(c["greedy"][i]["finish"] == "length" and same(greedy[i], gold[i]) for i in ids)},
    }
    # Where the fixed rule and the answer line disagree on a finished reply, and the answer line is right.
    alt = {i: answer_line(c["greedy"][i]["text"]) for i in ids}
    sec1["diagnostic_answer_line"] = {
        "greedy_differs": [i for i in ids if c["greedy"][i]["finish"] == "stop" and alt[i] is not None
                           and alt[i] != greedy[i]],
        "greedy_rule_wrong_line_right": [i for i in ids if not same(greedy[i], gold[i]) and same(alt[i], gold[i])],
        "samples_rule_wrong_line_right": sum(not same(extract(s["text"]), gold[i]) and same(answer_line(s["text"]), gold[i])
                                             for i in ids for s in c["samples"][i]["samples"]),
        "samples_rule_right_line_wrong": sum(same(extract(s["text"]), gold[i]) and not same(answer_line(s["text"]), gold[i])
                                             and answer_line(s["text"]) is not None
                                             for i in ids for s in c["samples"][i]["samples"]),
    }

    # Section 2: self-consistency.
    single = [acc({i: samp[i][k] for i in ids}) for k in range(N_SAMPLES)]
    vote_prefix, vote_mean, pass_mean, best_prefix, best_mean, wvote_mean = {}, {}, {}, {}, {}, {}
    for n in range(1, N_SAMPLES + 1):
        subsets = list(itertools.combinations(range(N_SAMPLES), n))
        vote_prefix[n] = acc({i: vote(samp[i][:n]) for i in ids})
        best_prefix[n] = acc({i: best_of(samp[i][:n], logodds[i][:n]) for i in ids})
        vm = pm = bm = wm = 0.0
        for sub in subsets:
            vm += acc({i: vote([samp[i][k] for k in sub]) for i in ids})
            bm += acc({i: best_of([samp[i][k] for k in sub], [logodds[i][k] for k in sub]) for i in ids})
            wm += acc({i: weighted_vote([samp[i][k] for k in sub], [score[i][k] for k in sub]) for i in ids})
            pm += sum(any(same(samp[i][k], gold[i]) for k in sub) for i in ids) / N
        vote_mean[n], best_mean[n], wvote_mean[n], pass_mean[n] = (x / len(subsets) for x in (vm, bm, wm, pm))
    vote5 = {i: vote(samp[i][:5]) for i in ids}
    vote8 = {i: vote(samp[i]) for i in ids}

    agree = {}
    for i in ids:
        top = vote(samp[i])
        agree[i] = sum(a == top for a in samp[i]) if top is not None else 0
    by_agree = {}
    for lo, hi in ((8, 8), (6, 7), (4, 5), (1, 3)):
        sel = [i for i in ids if lo <= agree[i] <= hi]
        by_agree[f"{lo}-{hi}" if lo != hi else str(lo)] = {
            "problems": len(sel), "vote8_right": sum(same(vote8[i], gold[i]) for i in sel)}

    sec2 = {
        "single_sample_acc": single, "single_sample_mean": sum(single) / N_SAMPLES,
        "vote_prefix": vote_prefix, "vote_mean_over_subsets": vote_mean,
        "vote5_acc": acc(vote5), "vote5_minus_greedy": acc(vote5) - acc(greedy),
        "greedy_vs_vote5": paired(greedy, vote5), "greedy_vs_vote8": paired(greedy, vote8),
        "majority_size_of_8": by_agree,
        "samples_hit_256": sum(s["finish"] == "length" for i in ids for s in c["samples"][i]["samples"]),
        "samples_finished_right": sum(s["finish"] == "stop" and same(a, gold[i]) for i in ids
                                      for s, a in zip(c["samples"][i]["samples"], samp[i])),
        "samples_cut_right": sum(s["finish"] == "length" and same(a, gold[i]) for i in ids
                                 for s, a in zip(c["samples"][i]["samples"], samp[i])),
        "samples_no_number": sum(a is None for i in ids for a in samp[i]),
        "sample_mean_tokens": sum(sum(stok[i]) for i in ids) / (N * N_SAMPLES),
        "distinct_answers_mean": sum(len({a for a in samp[i] if a is not None}) for i in ids) / N,
    }

    # Section 3: the model grades its own samples.
    flat = [(logodds[i][k], same(samp[i][k], gold[i])) for i in ids for k in range(N_SAMPLES)]
    prob = [(score[i][k], same(samp[i][k], gold[i])) for i in ids for k in range(N_SAMPLES)]
    pos = [s for s, r in flat if r]
    neg = [s for s, r in flat if not r]
    auc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    tops: dict[str, int] = {}
    for i in ids:
        for v in c["verify"][i]["verdicts"]:
            tops[v["top"]] = tops.get(v["top"], 0) + 1
    sec3 = {
        "best_prefix": best_prefix, "best_mean_over_subsets": best_mean,
        "weighted_vote_mean_over_subsets": wvote_mean, "pass_at_n_mean_over_subsets": pass_mean,
        "samples_right": len(pos), "samples_wrong": len(neg),
        "score_mean_right": sum(s for s, r in prob if r) / len(pos),
        "score_mean_wrong": sum(s for s, r in prob if not r) / len(neg),
        "logodds_median_right": sorted(pos)[len(pos) // 2], "logodds_median_wrong": sorted(neg)[len(neg) // 2],
        "said_yes_right": sum(s > 0 for s in pos), "said_yes_wrong": sum(s > 0 for s in neg),
        "score_rounds_to_1": sum(s >= 1.0 for s, _ in prob),
        "auc": auc, "top_token": dict(sorted(tops.items(), key=lambda kv: -kv[1])),
        "verify_mean_prompt_tokens": sum(v["prompt_tokens"] for i in ids for v in c["verify"][i]["verdicts"]) / (N * N_SAMPLES),
    }

    # Accuracy against tokens generated per problem (mean over problems).
    gen_tokens = {n: sum(sum(stok[i][:n]) for i in ids) / N for n in range(1, N_SAMPLES + 1)}
    cost = {
        "direct": {"tokens": sec1["direct_mean_tokens"], "acc": sec1["direct_acc"]},
        "greedy": {"tokens": sec1["greedy_mean_tokens"], "acc": sec1["greedy_acc"]},
        "vote": [{"n": n, "tokens": gen_tokens[n], "acc": vote_mean[n]} for n in vote_mean],
        "best_of_n": [{"n": n, "tokens": gen_tokens[n] + n, "read_tokens": n * sec3["verify_mean_prompt_tokens"],
                       "acc": best_mean[n]} for n in best_mean],
    }

    examples = []
    for i in ids:
        examples.append({"id": i, "gold": gold[i], "direct": direct[i], "greedy": greedy[i],
                         "samples": samp[i], "scores": score[i], "logodds": logodds[i]})
    chunks = read_jsonl(GEN / "chunks.jsonl")   # one row per generation chunk, as it ran
    gen_log = {"chunks": len(chunks), "minutes": round(sum(r["minutes"] for r in chunks), 1),
               "longest_chunk_minutes": max(r["minutes"] for r in chunks),
               "peak_rss_gb": max(r["peak_rss_gb"] for r in chunks),
               "peak_metal_gb": max(r["peak_metal_gb"] for r in chunks),
               "generated_tokens": sum(r["generated_tokens"] for r in chunks)}
    return {"generation_log": gen_log, "model": f"{MODEL_ID}@{MODEL_REVISION}", "weights": "4-bit (MLX), thinking off",
            "dataset": f"{DATASET_ID}@{DATASET_REVISION}, test rows 0-{N - 1}",
            "n_problems": N, "n_samples": N_SAMPLES, "temperature": 0.7,
            "section1": sec1, "section2": sec2, "section3": sec3, "cost": cost, "per_problem": examples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    guard = Guard("analyse")
    out = analyse(load())
    out["guard"] = guard.report()
    s1, s2, s3 = out["section1"], out["section2"], out["section3"]
    print(f"answer only {s1['direct_acc']:.1%}  greedy CoT {s1['greedy_acc']:.1%}  "
          f"vote(5) {s2['vote5_acc']:.1%}  vote(8) {s2['vote_prefix'][8]:.1%}  "
          f"best-of-8 {s3['best_prefix'][8]:.1%}  any of 8 {s3['pass_at_n_mean_over_subsets'][8]:.1%}")
    write_results("analysis", out)
    if a.figures:
        import figures

        figures.main()


if __name__ == "__main__":
    main()
