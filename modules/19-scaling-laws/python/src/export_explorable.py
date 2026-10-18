"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/19-scaling-laws.html) holds one generated
block between the BEGIN DATA and END DATA markers. This script rewrites that
block and nothing else, so no number on the page is typed by hand.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from common import MODULE, SIZES, STEPS, read_results, read_runs

PAGE = MODULE.parents[1] / "explorables" / "19-scaling-laws.html"
BEGIN = "/* BEGIN DATA: written by modules/19-scaling-laws/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    runs = read_runs()
    fit = read_results("fit")
    fr = read_results("frontier")
    sizes = []
    for d in SIZES:
        r = runs[d]
        s0 = next(x for x in r["runs"] if x["seed"] == 0)
        secs = r["guard"]["stages_seconds"]["train seed 0"]
        sizes.append({
            "d": d, "n": s0["n_non_embedding"], "total": s0["n_params_total"],
            "seeds": [{"seed": x["seed"], "curve": [[c["tokens"], round(c["held_loss"], 6)] for c in x["curve"]]}
                      for x in r["runs"]],
            "train_seconds": secs,
            "flops_per_s": round(6 * s0["n_non_embedding"] * s0["curve"][-1]["tokens"] / secs),
            "sample": r["sample_seed0"],
        })
    lp = {k: fit["l_of_d"][str(d)] for k, d in (("a", fit["l_of_d_largest_two"][0]), ("b", fit["l_of_d_largest_two"][1]))}
    return {
        "steps": STEPS, "tokens": runs[SIZES[0]]["runs"][0]["curve"][-1]["tokens"],
        "warmup_tokens": runs[SIZES[0]]["warmup"] * runs[SIZES[0]]["batch"] * runs[SIZES[0]]["block"],
        "sizes": sizes,
        "fit": {
            "alpha": round(fit["l_of_n_main"]["exponent"], 4), "a": round(fit["l_of_n_main"]["a"], 4),
            "r2": round(fit["l_of_n_main"]["r2"], 4), "band": round(fit["band"], 4),
            "alpha_by_tokens": {k: round(v["exponent"], 4) for k, v in fit["l_of_n"].items()},
            "l_of_d": {str(fit["l_of_d_largest_two"][i]): {"beta": round(v["exponent"], 4), "r2": round(v["r2"], 3)}
                       for i, v in enumerate(lp.values())},
            "joint": {k: fit["joint"][k] for k in ("E", "A", "B", "alpha", "beta", "rmse", "max_abs_err", "n_points")},
            "baselines": {k: round(v, 4) for k, v in fit["baselines"].items()},
        },
        "frontier": {
            "rows": [{"C": r["C"], "d": r["best_d"], "loss": round(r["loss"], 4)} for r in fr["rows"]],
            "joint_n_opt_exponent": round(fr["joint_n_opt_exponent"], 3),
        },
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
