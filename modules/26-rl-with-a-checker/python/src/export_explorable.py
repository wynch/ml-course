"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/26-rl-with-a-checker.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no measured number on the page is
typed by hand.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import common
import task

PAGE = common.MODULE.parents[1] / "explorables" / "26-rl-with-a-checker.html"
BEGIN = "/* BEGIN DATA: written by modules/26-rl-with-a-checker/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"
ARMS = ["grpo", "drgrpo", "rloo", "random", "random_clip"]


def run(name: str) -> dict:
    return json.loads((common.RESULTS / "runs" / f"{name}.json").read_text())


def build() -> dict:
    R = common.read_results("rl")
    base = run("base_eval")["final"]
    out = {
        "test_problems": R["test_problems"],
        "n_samples": base["n_samples"],
        "base": {"right": base["right_per_problem"], "pass_at_k": base["pass_at_k"],
                 "worked_share": base["worked_share"], "examples": base["examples"]},
        "arms": {},
    }
    for arm in ARMS:
        s0 = run(f"{arm}_s0")
        A = R["arms"][arm]
        out["arms"][arm] = {
            "right_s0": s0["final"]["right_per_problem"],
            "examples_s0": s0["final"]["examples"],
            "pass_at_k": {k: v["mean"] for k, v in A["pass_at_k"].items()},
            "steps": A["curve_steps"],
            "reward": [round(x, 4) for x in A["curve_mean"]["reward"]],
            "worked_share": [round(x, 4) for x in A["curve_mean"]["worked_share"]],
            "kl": [round(x, 4) for x in A["curve_mean"]["kl_to_base"]],
        }
    pairs = task.splits()[0][: len(base["right_per_problem"])]
    out["pairs"] = pairs.tolist()
    if (common.RESULTS / "library_after.json").exists():
        b, a = common.read_results("library_base"), common.read_results("library_after")
        out["library"] = {"before": b["measure"]["pass_at_k"], "after": a["measure"]["pass_at_k"],
                          "greedy_before": b["measure"]["greedy_acc"], "greedy_after": a["measure"]["greedy_acc"]}
    return out


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
    b = block()
    new = pattern.sub(lambda _: b, page)
    if args.check:
        if new != page:
            sys.exit(f"{PAGE.name} is stale: run src/export_explorable.py")
        print(f"{PAGE.name} is in sync with results/")
        return
    PAGE.write_text(new)
    print(f"wrote the data block of {PAGE.name} ({len(b) // 1024} KB)")


if __name__ == "__main__":
    main()
