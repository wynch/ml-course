"""Reads results/runs/*.json and writes results/rl.json: every number the README quotes
about the from-scratch runs, means and ranges over the 3 seeds, and each written
expectation scored against them. Runs no model; well under a second.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import task
from common import RESULTS, write_results

RUNS = RESULTS / "runs"
ARMS = ["grpo", "drgrpo", "rloo", "random", "random_clip"]
SEEDS = [0, 1, 2]
TRUE_ARMS = ["grpo", "drgrpo", "rloo"]


def load(name: str) -> dict:
    return json.loads((RUNS / f"{name}.json").read_text())


def stat(xs) -> dict:
    xs = [float(x) for x in xs]
    return {"mean": float(np.mean(xs)), "min": float(min(xs)), "max": float(max(xs)), "seeds": xs}


def window_mean(log: list[dict], key: str, lo: int = 100, hi: int = 290) -> float:
    vals = [r[key] for r in log if lo <= r["step"] <= hi and r[key] is not None]
    return float(np.mean(vals))


def crossover(base_right: list[int], rl_right: list[int]) -> dict:
    """Problems the base solves at least once in 128 and the RL model never, and the reverse."""
    b = np.array(base_right) > 0
    r = np.array(rl_right) > 0
    return {"base_only": int((b & ~r).sum()), "rl_only": int((r & ~b).sum()),
            "neither": int((~b & ~r).sum())}


def analyse() -> dict:
    base = load("base_eval")["final"]
    out = {"base": {k: v for k, v in base.items() if k not in ("right_per_problem",)}, "arms": {}}
    for arm in ARMS:
        runs = [load(f"{arm}_s{s}") for s in SEEDS]
        fin = [r["final"] for r in runs]
        ks = list(base["pass_at_k"])
        out["arms"][arm] = {
            "pass_at_k": {k: stat([f["pass_at_k"][k] for f in fin]) for k in ks},
            "pass_at_k_minus_base": {k: stat([f["pass_at_k"][k] - base["pass_at_k"][k] for f in fin]) for k in ks},
            "greedy_acc": stat([f["greedy_acc"] for f in fin]),
            "worked_share": stat([f["worked_share"] for f in fin]),
            "mean_length": stat([f["mean_length"] for f in fin]),
            "acc_direct": stat([f["acc_direct"] for f in fin if f["acc_direct"] is not None]) if any(
                f["acc_direct"] is not None for f in fin) else None,
            "acc_worked": stat([f["acc_worked"] for f in fin if f["acc_worked"] is not None]) if any(
                f["acc_worked"] is not None for f in fin) else None,
            "distinct_answers_mean": stat([f["distinct_answers_mean"] for f in fin]),
            "kl_final": stat([r["log"][-1]["kl_to_base"] for r in runs]),
            "length_wrong_100_290": stat([window_mean(r["log"], "length_wrong") for r in runs]),
            "length_right_100_290": stat([window_mean(r["log"], "length_right") for r in runs]),
            "dead_groups_100_290": stat([window_mean(r["log"], "dead_groups") for r in runs]),
            "crossover": [crossover(base["right_per_problem"], f["right_per_problem"]) for f in fin],
            "minutes": stat([r["guard"]["minutes"] for r in runs]),
            "peak_rss_gb": max(r["guard"]["peak_rss_gb"] for r in runs),
            "curve_mean": {
                key: [float(np.mean([r["log"][i][key] for r in runs])) for i in range(len(runs[0]["log"]))]
                for key in ("reward", "length", "worked_share", "kl_to_base", "dead_groups")
            },
            "curve_steps": [row["step"] for row in runs[0]["log"]],
            "test_curve_mean": [float(np.mean([r["test_curve"][i]["sample_acc_first_200"] for r in runs]))
                                for i in range(len(runs[0]["test_curve"]))],
            "test_curve_steps": [row["step"] for row in runs[0]["test_curve"]],
        }
    A = out["arms"]
    b1 = base["pass_at_k"]["1"]
    exp = {}
    exp["E1"] = all(x >= 0.85 for a in TRUE_ARMS for x in A[a]["pass_at_k"]["1"]["seeds"])
    exp["E2"] = abs(A["rloo"]["pass_at_k"]["1"]["mean"] - A["drgrpo"]["pass_at_k"]["1"]["mean"]) <= 0.02
    exp["E3"] = A["grpo"]["length_wrong_100_290"]["mean"] - A["drgrpo"]["length_wrong_100_290"]["mean"] >= 1.0
    exp["E4"] = all(x > 0.9 for a in TRUE_ARMS for x in A[a]["worked_share"]["seeds"])
    exp["E5"] = all(x >= -0.005 for a in TRUE_ARMS for k in A[a]["pass_at_k_minus_base"]
                    for x in A[a]["pass_at_k_minus_base"][k]["seeds"]) and all(
        c["base_only"] <= 5 for a in TRUE_ARMS for c in A[a]["crossover"])
    exp["E6"] = (all(abs(x - b1) <= 0.05 for x in A["random"]["pass_at_k"]["1"]["seeds"])
                 and all(0.15 <= x <= 0.35 for x in A["random"]["worked_share"]["seeds"])
                 and all(x < 0.3 for x in A["random"]["kl_final"]["seeds"]))
    exp["E7"] = (all(x < base["worked_share"] for x in A["random_clip"]["worked_share"]["seeds"])
                 and all(-0.10 <= x - b1 <= 0.0 for x in A["random_clip"]["pass_at_k"]["1"]["seeds"]))
    exp["E8"] = all(x > 1.0 for a in TRUE_ARMS for x in A[a]["kl_final"]["seeds"])
    out["expectations_met"] = exp
    out["expectations_text"] = json.loads((RESULTS / "expectations.json").read_text())["expectations"]
    out["rloo_over_drgrpo_advantage"] = 8 / 7
    # Adam divides each step by a running gradient size, so a constant factor on the
    # advantages (RLOO = Dr. GRPO x 8/7) changes almost nothing: same samples, same batches.
    same = []
    for s in SEEDS:
        a, b = load(f"drgrpo_s{s}")["log"], load(f"rloo_s{s}")["log"]
        diff = [x["step"] for x, y in zip(a, b) if x["reward"] != y["reward"]]
        same.append({"logged_steps": len(a), "same_batch_reward": len(a) - len(diff),
                     "first_different_step": diff[0] if diff else None})
    out["rloo_vs_drgrpo"] = same
    if (RESULTS / "library_after.json").exists():
        lb = json.loads((RESULTS / "library_base.json").read_text())["measure"]
        la = json.loads((RESULTS / "library_after.json").read_text())["measure"]
        tr = json.loads((RESULTS / "library_train.json").read_text())
        hist = [h for h in tr["log_history"] if "reward" in h]
        out["library"] = {
            "pass_at_k_before": lb["pass_at_k"], "pass_at_k_after": la["pass_at_k"],
            "greedy_before": lb["greedy_acc"], "greedy_after": la["greedy_acc"],
            "reply_tokens_before": lb["mean_reply_tokens"], "reply_tokens_after": la["mean_reply_tokens"],
            "crossover_in_16": crossover(lb["right_per_problem"], la["right_per_problem"]),
            "train_reward_first_log": hist[0]["reward"], "train_reward_last_log": hist[-1]["reward"],
            "train_reward_last_20_steps": float(np.mean([h["reward"] for h in hist if h["step"] > 100])),
            "steps": tr["config"]["steps"],
            "train_minutes": sum(c["guard"]["minutes"] for c in tr["chunks"]),
        }
    out["test_problems"] = task.N_TEST
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    out = analyse()
    write_results("rl", out)
    for k, v in out["expectations_met"].items():
        print(k, "met" if v else "MISSED", "-", out["expectations_text"][k])
    if args.figures:
        import figures

        figures.main()


if __name__ == "__main__":
    main()
