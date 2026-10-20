"""Section 1, follow-up: give the shallowest shape its own learning rate.

Not pre-registered. Written after section 1's results, which showed a gap in
its protocol: the rate was tuned on the middle shape only, yet the shallowest
shape (one block, wide) won four of the eight cells at a rate tuned for
another shape, and in the 4-bit cell at 1.5 MB it lost badly at that rate.
So here, in every arm and budget, the shallowest shape runs the same rate
grid (with the same step beyond an end), seed 0; its best rate then runs
seeds 1 and 2. A cell's follow-up winner is section 1's winner or the tuned
shallow shape, whichever scores lower on the tune half; the report half
scores it, as before.

Runs are cached in results/followup_runs.jsonl; section 1's runs are reused
when they match. Chunked like matched.py.

Run me (after matched.py is done):
    uv run python src/followup.py                     # one chunk; repeat until it says "done"
"""

from __future__ import annotations

import argparse
import json
import time

import common  # noqa: F401  (MPS guard before torch)
import numpy as np

from chars import splits
from common import RESULTS, Guard, get_device, read_results, write_results
from matched import (ARMS, BUDGETS, EVAL_RESERVE, LR_BEYOND, LR_GRID, SEEDS, arm_bytes, evaluate, key, kind_of,
                     partial_path, shapes)
from train import train

FOLLOW = RESULTS / "followup_runs.jsonl"


def load(path) -> dict:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["key"]: r for r in rows if "key" in r}


def plan(vocab: int, runs: dict):
    todo = []
    for budget in BUDGETS:
        for arm in ARMS:
            kind = kind_of(arm)
            tag = arm if kind == "float" else "ternary"
            d, L = shapes(arm, budget, vocab)[2]

            def score(lr, seed=0):
                r = runs.get(key(kind, d, L, lr, seed))
                return None if r is None else r["losses"][tag]["tune"]

            grid = list(LR_GRID)
            missing = [lr for lr in grid if score(lr) is None]
            if missing:
                todo += [(arm, budget, d, L, lr, 0) for lr in missing]
                continue
            best = min(grid, key=score)
            if best in LR_BEYOND:
                if score(LR_BEYOND[best]) is None:
                    todo.append((arm, budget, d, L, LR_BEYOND[best], 0))
                    continue
                grid.append(LR_BEYOND[best])
                best = min(grid, key=score)
            todo += [(arm, budget, d, L, best, s) for s in SEEDS[1:] if score(best, s) is None]
    return todo


def summarise(vocab: int, runs: dict) -> dict:
    m = read_results("matched")
    cells = []
    for c in m["cells"]:
        arm, budget = c["arm"], c["budget_mb"] * 1e6
        kind = kind_of(arm)
        tag = arm if kind == "float" else "ternary"
        d, L = shapes(arm, budget, vocab)[2]
        tried = sorted((r for r in runs.values() if r["kind"] == kind and (r["d"], r["n_layer"]) == (d, L)
                        and r["seed"] == 0), key=lambda r: r["lr"])
        best = min(tried, key=lambda r: r["losses"][tag]["tune"])
        seeds = sorted((r for r in runs.values() if r["kind"] == kind and (r["d"], r["n_layer"], r["lr"])
                        == (d, L, best["lr"])), key=lambda r: r["seed"])
        shallow = {"d": d, "n_layer": L, "lr": best["lr"], "bytes": arm_bytes(arm, d, L, vocab),
                   "tune_seed0": best["losses"][tag]["tune"],
                   "rates": [{"lr": r["lr"], "tune": r["losses"][tag]["tune"], "report": r["losses"][tag]["report"]}
                             for r in tried],
                   "report_by_seed": [r["losses"][tag]["report"] for r in seeds],
                   "before_rounding_report": [r["losses"]["fp32"]["report"] for r in seeds] if kind == "float" else None}
        shallow["report_mean"] = round(float(np.mean(shallow["report_by_seed"])), 4)
        shallow["report_sd"] = round(float(np.std(shallow["report_by_seed"], ddof=1)), 4)
        s1_tune = min(t["tune"] for t in c["tried"])
        winner = "shallow, own rate" if shallow["tune_seed0"] < s1_tune else "section 1"
        w = shallow if winner != "section 1" else {**c["best"], "bytes": c["bytes"], "report_by_seed": c["report_by_seed"],
                                                   "report_mean": c["report_mean"], "report_sd": c["report_sd"],
                                                   "before_rounding_report": c.get("before_rounding_report")}
        cells.append({"budget_mb": c["budget_mb"], "arm": arm, "shallow": shallow,
                      "section1": {**c["best"], "report_mean": c["report_mean"], "tune_seed0": s1_tune},
                      "winner": winner,
                      "best": {"d": w["d"], "n_layer": w["n_layer"], "lr": w["lr"], "bytes": w["bytes"],
                               "report_by_seed": w["report_by_seed"], "report_mean": w["report_mean"],
                               "report_sd": w["report_sd"], "before_rounding_report": w["before_rounding_report"]}})
    return {"cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=4.5)
    args = ap.parse_args()

    guard = Guard("followup.py")
    device = get_device()
    train_ids, tune, report, vocab = splits()
    own = load(FOLLOW)
    runs = {**load(RESULTS / "matched_runs.jsonl"), **own}
    todo = plan(vocab.size, runs)
    t0 = time.perf_counter()
    deadline = t0 + args.max_minutes * 60 - EVAL_RESERVE
    trained = 0
    while todo and time.perf_counter() < deadline:
        arm, budget, d, L, lr, seed = todo[0]
        kind = kind_of(arm)
        k = key(kind, d, L, lr, seed)
        model, info = train(kind, d, L, lr, seed, train_ids, vocab.size, device, ckpt=partial_path(k), deadline=deadline)
        if model is None:
            print(f"  {k}: paused at the chunk's end, saved to resume")
            break
        row = {"key": k, **info, "losses": evaluate(model, kind, tune, report, device), "for": ["followup", arm, budget]}
        with FOLLOW.open("a") as f:
            f.write(json.dumps(row) + "\n")
        runs[k] = row
        trained += 1
        tag = arm if kind == "float" else "ternary"
        print(f"  {k:32s} {info['seconds']:6.1f} s  tune {row['losses'][tag]['tune']:.4f}  ({arm}, {budget/1e6} MB)")
        guard.tick(k)
        del model
        common.free()
        todo = plan(vocab.size, runs)
    g = guard.report()
    with FOLLOW.open("a") as f:
        f.write(json.dumps({"chunk": g, "runs_trained": trained}) + "\n")
    if todo:
        print(f"not done: {len(todo)} runs queued next; call again")
        return
    out = summarise(vocab.size, runs)
    chunks = [json.loads(line)["chunk"] for line in FOLLOW.read_text().splitlines() if '"chunk"' in line]
    out["chunks"] = {"count": len(chunks), "longest_minutes": max(c["minutes"] for c in chunks),
                     "total_minutes": round(sum(c["minutes"] for c in chunks), 2),
                     "peak_rss_gb": max(c["peak_rss_gb"] for c in chunks),
                     "peak_mps_driver_gb": max(c["peak_mps_driver_gb"] for c in chunks)}
    out["guard"] = g
    write_results("followup", out)
    print("done")


if __name__ == "__main__":
    main()
