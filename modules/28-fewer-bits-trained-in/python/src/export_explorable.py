"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/28-fewer-bits-trained-in.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else. One number on the page is typed by hand:
section 2's g = 0.0821, mean |W| of the qkv matrix of the cached 0.5 MB ternary
model (0.08206, read from its checkpoint, which export.py trains).

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "28-fewer-bits-trained-in.html"
BEGIN = "/* BEGIN DATA: written by modules/28-fewer-bits-trained-in/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    m = read_results("matched")
    e = read_results("export")
    k = read_results("kv_hand")
    lib = read_results("kv_mlx")
    f = read_results("followup")
    vocab, chars = m["setup"]["vocab"], m["setup"]["chars_seen"]

    def params(d, L):
        return 12 * d * d * L + (vocab + m["setup"]["block"]) * d + 13 * d * L + 2 * d

    follow = []
    for c in f["cells"]:
        b = c["best"]
        follow.append({"budget_mb": c["budget_mb"], "arm": c["arm"], "winner": c["winner"],
                       "best": {"d": b["d"], "n_layer": b["n_layer"], "lr": b["lr"]}, "bytes": b["bytes"],
                       "linear_weights": 12 * b["d"] ** 2 * b["n_layer"],
                       "chars_per_param": round(chars / params(b["d"], b["n_layer"]), 2),
                       "report_by_seed": b["report_by_seed"], "report_mean": b["report_mean"], "report_sd": b["report_sd"],
                       "before": b["before_rounding_report"],
                       "tried": [{"d": c["shallow"]["d"], "n_layer": c["shallow"]["n_layer"], "lr": x["lr"],
                                  "bytes": c["shallow"]["bytes"], "report": x["report"]} for x in c["shallow"]["rates"]]})
    return {
        "matched": {
            "setup": {key: m["setup"][key] for key in ("budgets_mb", "lr_grid", "steps", "batch", "block",
                                                         "chars_seen", "group", "ternary_bits", "vocab")},
            "cells": [{key: c[key] for key in ("budget_mb", "arm", "best", "bytes", "params", "linear_weights",
                                               "chars_per_param", "report_by_seed", "report_mean", "report_sd",
                                               "tried")}
                      | {"before": c.get("before_rounding_report"), "export": c.get("export")}
                      for c in m["cells"]],
            "follow": follow,
        },
        "export": {"base": e["base"], "steps": e["finetune"]["steps"], "rows": e["rows"]},
        "kv": {
            "rows": [{key: r[key] for key in ("bits", "keys", "rotation", "ppl")} for r in k["rows"]],
            "tokens": k["ppl_tokens"], "outliers": k["outliers"]["per_layer"], "sample": k["outliers"]["sample"],
        },
        "lib": {"rows": [{key: r[key] for key in ("bits", "ppl", "hand_ppl")} for r in lib["rows"]],
                "bytes": lib["cache_bytes"], "speed": lib["speed"], "speed_setup": lib["speed_setup"]},
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
