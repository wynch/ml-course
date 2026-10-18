"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/23-compression.html) holds one generated
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

from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "23-compression.html"
BEGIN = "/* BEGIN DATA: written by modules/23-compression/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    d = read_results("distil")
    p = read_results("prune")
    q = read_results("quant")
    m = read_results("mlx_way")
    sm = q["shown_matrix"]
    return {
        "distil": {
            "teacher": d["teacher"], "student": d["student"], "steps": d["setup"]["steps"],
            "T": d["setup"]["T"], "alpha": d["setup"]["alpha"], "mean": d["mean_val_loss"],
            "gaps": d["gap_scratch_minus_distil"],
            "runs": [{"kind": r["kind"], "seed": r["seed"], "final": r["val_loss"],
                      "curve": [[c["step"], c["val_loss"]] for c in r["curve"]]} for r in d["runs"]],
        },
        "prune": {
            "tokens": p["ppl_setup"]["predicted_tokens"], "request": p["request"],
            "rows": [{"method": r["method"], "f": r["fraction"], "ppl": r["ppl"], "reply": r["reply"]}
                     for r in p["rows"]],
        },
        "quant": {
            "rows": [{"scheme": r["scheme"], "ppl": r["ppl"], "bytes": r["bytes"], "bits": r["bits_per_weight"],
                      "err": r["mean_rel_error"]} for r in q["rows"]],
            "counts": {"quantized": q["quantized_weights"], "kept_bf16": q["kept_bf16"]},
            "matrix": {"name": sm["name"], "shape": sm["shape"], "rows": sm["rows"], "cols": sm["cols"],
                       "b64": sm["bf16_b64"]},
        },
        "mlx": {
            "rows": [{"setting": r["setting"], "ppl": r["ppl"], "tps": r["speed"]["median_tok_s"]} for r in m["rows"]],
            "torch": [{"scheme": r["scheme"], "ppl": r["ppl"], "tps": r.get("speed", {}).get("median_tok_s")}
                      for r in q["rows"]],
            "check": m["rows"][1]["check_vs_hand"], "file_bytes": m["safetensors_bytes"],
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
