"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/24-reinforcement-learning.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no number on the page is typed by
hand.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

import cartpole as cp
from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "24-reinforcement-learning.html"
BEGIN = "/* BEGIN DATA: written by modules/24-reinforcement-learning/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def r1(xs, nd=1):
    return [None if x is None else round(float(x), nd) for x in xs]


def build() -> dict:
    b = read_results("bandits")
    q = read_results("qlearning")
    rf = read_results("reinforce")
    pp = read_results("ppo")
    ppo_runs = {k: read_results(k) for k in ("ppo", "ppo_fast", "ppo_fast_noclip")}

    grid = b["grid"]
    keep = [i for i, g in enumerate(grid) if (g <= 100 and g % 5 == 0) or g % 100 == 0 or g == 1]
    rules = {}
    for k, v in b["rules"].items():
        a = np.array(v["regret"])[:, keep]
        rules[k] = {"med": r1(np.median(a, 0)), "lo": r1(a.min(0)), "hi": r1(a.max(0)),
                    "final_median": v["final_median"], "final_min": v["final_min"], "final_max": v["final_max"],
                    "share": v["best_arm_share_last_1000_median"]}
    steps = np.array([r["steps"] for r in q["runs"]])
    run0 = q["runs"][0]

    def first500(r):
        return [x["first_500_at_step"] for x in r["runs"]]

    def ppo_view(r):
        log = [x["log"] for x in r["runs"]]
        return {"lr": r["lr"], "use_clip": r["use_clip"], "curves": [x["curve"] for x in r["runs"]],
                "eval": [x["eval_greedy_mean"] for x in r["runs"]], "first500": first500(r),
                "clip_steps": [x["step"] for x in log[0]],
                "clip_seeds": [[x["clip_frac"] for x in L] for L in log],
                "clip_mean": [round(float(np.mean([x["clip_frac"] for x in L])), 4) for L in log]}

    return {
        "bandit": {"means": b["means"], "pulls": b["pulls"], "seeds": b["seeds"], "epsilon": b["epsilon"],
                   "grid": [grid[i] for i in keep], "rules": rules,
                   "random_slope": b["random_expected_slope"], "egreedy_slope": b["egreedy_expected_late_slope"],
                   "lai_c": b["lai_robbins_constant"], "lai_at": b["lai_robbins_at_pulls"]},
        "q": {"size": q["size"], "start": q["start"], "goal": q["goal"], "walls": q["walls"], "shortest": q["bfs_shortest"],
              "episodes": q["episodes"], "seeds": q["seeds"], "epsilon": q["epsilon"], "alpha": q["alpha"],
              "med": [int(x) for x in np.median(steps, 0).round()], "lo": steps.min(0).tolist(), "hi": steps.max(0).tolist(),
              "V": run0["V"], "path": run0["path"], "dist": q["bfs_dist"],
              "last50_med": q["summary"]["last50_mean_steps_median"], "equal_bfs": q["summary"]["greedy_path_equals_bfs"]},
        "cp": {"max_steps": cp.MAX_STEPS, "x_limit": cp.X_LIMIT, "theta_limit": round(cp.THETA_LIMIT, 6),
               "half_length": cp.HALF_LENGTH, "tau": cp.TAU,
               "replays": {"untrained": {"states": pp["untrained_replay"], "len": pp["untrained_len"]},
                           "reinforce": {"states": rf["runs"][0]["replay"], "len": rf["runs"][0]["replay_len"]},
                           "ppo": {"states": pp["runs"][0]["replay"], "len": pp["runs"][0]["replay_len"]}},
               "train": {"grid": rf["grid"], "budget": rf["budget_steps"],
                         "reinforce": {"curves": [x["curve"] for x in rf["runs"]],
                                       "eval": [x["eval_greedy_mean"] for x in rf["runs"]], "first500": first500(rf)},
                         "ppo": {k: ppo_view(v) for k, v in ppo_runs.items()},
                         "batch": pp["n_envs"] * pp["n_steps"], "epochs": pp["epochs"], "clip": pp["clip"]}},
    }


def block() -> str:
    data = json.dumps(build(), ensure_ascii=False, separators=(",", ":"))
    return f"{BEGIN}\nconst DATA = {data};\n{END}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    page = PAGE.read_text()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(page):
        sys.exit(f"markers not found in {PAGE}")
    new = pattern.sub(lambda _: block(), page)
    if args.check:
        if new != page:
            sys.exit(f"{PAGE.name} is stale: run src/export_explorable.py")
        print(f"{PAGE.name} is in sync with results/")
        return
    PAGE.write_text(new)
    print(f"wrote the data block of {PAGE.name} ({len(block()) // 1024} KB)")


if __name__ == "__main__":
    main()
