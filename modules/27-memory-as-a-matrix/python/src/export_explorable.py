"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/27-memory-as-a-matrix.html) holds one
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

from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "27-memory-as-a-matrix.html"
BEGIN = "/* BEGIN DATA: written by modules/27-memory-as-a-matrix/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    cap, mq, lib = read_results("capacity"), read_results("mqar"), read_results("library")
    grid = [{"rule": g["rule"], "d_head": g["d_head"], "state_bytes": g["state_bytes"],
             "mean": g["mean"], "spread": g["spread"]} for g in mq["grid"].values()]
    seeds = len(next(iter(mq["grid"].values()))["seeds"])
    hybrid = {k: {"mean": h["mean"], "spread": h["spread"]} for k, h in mq["hybrid"].items()}
    models = {}
    for key, m in lib["models"].items():
        c = m["config"]
        if m["layer_types"]:
            lin = sum(t == "linear_attention" for t in m["layer_types"])
            layers = f"{lin} Gated DeltaNet layers and {len(m['layer_types']) - lin} attention layers"
        else:
            layers = f"{c['num_hidden_layers']} attention layers"
        models[key] = {
            "layers": layers,
            "bytes_per_token": lib["bytes_per_token"][key], "fixed_bytes": lib["fixed_bytes"][key],
            "rows": {str(T): {"cache_bytes": r["cache_bytes"], "prefill": r["prefill_s_median"],
                              "decode": r["decode_tokens_per_s_median"], "found": r["found"],
                              "trials": len(r["trials"])} for T, r in m["contexts"].items()},
        }
    last = str(mq["pairs"][-1])
    g = mq["grid"]
    biggest = max(x["d_head"] for x in g.values() if x["rule"] != "attention")
    lin, dl, gd = (g[f"{r}_dk{biggest}"]["mean"][last] for r in ("linear", "delta", "gated_delta"))
    q3, q35 = lib["models"]["qwen3"]["contexts"], lib["models"]["qwen3.5"]["contexts"]
    T = str(lib["contexts"][-1])
    ov = cap["overwrite"]
    notice = [
        "In section 1, pick the delta rule with β = 1 and write pair #4 (0° again, a new value): the curve passes "
        "through the new value at 0°, and pair #1 is gone. Switch to linear: the read at 0° adds the old and new values, plus a share of pair #3 at 45°.",
        "Still in section 1, write a third direction with the delta rule: one of the older pairs must blur. "
        "Two numbers hold two independent keys, never three.",
        f"In section 2, linear attention keeps every pair about equally well; the delta rule and decay keep the newest "
        f"and lose the oldest. With random keys, linear attention reads more pairs back right at large n "
        f"(module: {cap['curves']['linear']['recall'][cap['ns'].index(64)]:.2f} against "
        f"{cap['curves']['delta']['recall'][cap['ns'].index(64)]:.2f} at 64 pairs).",
        f"Overwrite one key with a new value: the delta rule's read matches the new value (cosine "
        f"{ov['delta']['cos_new']:.2f}); linear attention returns a blend (cosine {ov['linear']['cos_new']:.2f} with "
        f"the new value, {ov['linear']['cos_old']:.2f} with the old).",
        f"In section 3, at heads of {biggest} and {last} pairs: linear {lin:.2f}, delta {dl:.2f}, gated delta {gd:.2f}. "
        f"Compare the seed spreads in the table before reading anything into small gaps.",
        f"In section 4, slide the context: Qwen3-0.6B's cache grows by "
        f"{lib['bytes_per_token']['qwen3']:,.0f} bytes a token, Qwen3.5-0.8B's by {lib['bytes_per_token']['qwen3.5']:,.0f} "
        f"on top of a fixed {lib['fixed_bytes']['qwen3.5'] / 2**20:.1f} MB. At {int(T):,} tokens: "
        f"{q3[T]['cache_bytes'] / 2**30:.2f} GB against {q35[T]['cache_bytes'] / 2**20:.0f} MB.",
    ]
    return {
        "capacity": {"d_k": cap["d_k"], "rho": cap["rho"], "alpha": cap["alpha"], "ns": cap["ns"],
                     "curves": {k: {"recall": v["recall"]} for k, v in cap["curves"].items()},
                     "curves_correlated": {k: {"recall": v["recall"]} for k, v in cap["curves_correlated"].items()}},
        "mqar": {"pairs": mq["pairs"], "T": mq["T"], "d_model": mq["d_model"], "steps": mq["steps"], "seeds": seeds,
                 "grid": grid, "hybrid": hybrid},
        "library": {"contexts": lib["contexts"], "models": models},
        "notice": notice,
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
