"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/32-is-the-explanation-real.html) holds one
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

import numpy as np

import common

PAGE = common.MODULE.parents[1] / "explorables" / "32-is-the-explanation-real.html"
BEGIN = "/* BEGIN DATA: written by modules/32-is-the-explanation-real/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"
SCATTER_KEEP = 1500          # points per model in the scatter: the 1,500 largest |attribution|, then every 10th


def scatter(att, eff) -> list:
    a, e = np.array(att), np.array(eff)
    order = np.argsort(-np.abs(a))
    keep = np.r_[order[:SCATTER_KEEP], order[SCATTER_KEEP::10]]
    return [[round(float(a[i]), 3), round(float(e[i]), 3)] for i in keep]


def build() -> dict:
    D = common.read_results("dicts")
    B = common.read_results("browse")
    A = common.read_results("attribution")
    L = common.read_results("library")
    S, Dn = common.read_results("train_sparse"), common.read_results("train_dense")
    summ = {}
    for which, M in D["models"].items():
        summ[which] = {"true_alive": M["true_alive"], "true": {k: M["true_neurons"][k] for k in ("detection", "token_r2")}}
        for kind, K in M["dicts"].items():
            summ[which][kind] = {
                "mmcs": [K[f"s{s}"]["mmcs_value"] for s in (0, 1, 2)],
                "recovered": [K[f"s{s}"]["recovered_share"] for s in (0, 1, 2)],
                "fvu": K["s0"]["fvu"], "dead": K["s0"]["dead"],
                "detection": K["s0"]["detection"], "token_r2": K["s0"]["token_r2"],
                "best_cos_hist": K["s0"]["best_cos_hist"],
                "seeds": {k: K["seeds"][k] for k in ("shared_share_mean", "on_truth", "off_truth")},
            }
    att = {}
    for which, M in A["models"].items():
        att[which] = {
            "graph": M["graph"], "scatter": scatter(M["scatter"]["att"], M["scatter"]["eff"]),
            "r_pooled": M["pearson_r_pooled"], "r_median": M["pearson_r_median"],
            "faith": M["faithfulness_median"],
            "top10_beats": M["top10_beats_random_mean"], "ratio": M["median_ratio_top_vs_random"],
            "drop_top10": M["median_drop_top10"], "drop_random10": M["median_drop_random10"],
            "prompts": [{"target": r["target_piece"], "p": r["p_full"], "r": r["pearson_r"],
                         "top": r["drop_top10"], "rand": r["drop_random10_mean"]} for r in M["prompts"]],
        }
    return {
        "train": {"sparse": S["val"]["loss"], "dense": Dn["val"]["loss"], "alive": S["alive_per_layer_1e-5"]},
        "dense_neurons": {k: D["dense_neurons"][k] for k in ("detection", "token_r2")},
        "dicts": summ,
        "browse": B["latents"],
        "attribution": att,
        "n_prompts": A["n_prompts"],
        "library": {k: {kk: v[kk] for kk in ("fve", "l0", "detection", "token_r2", "n_latents", "scorable")}
                    for k, v in L["saes"].items()},
        "library_browse": L["browse"],
    }


def block(d: dict) -> str:
    return BEGIN + "\nconst DATA = " + json.dumps(d, ensure_ascii=False, separators=(",", ":")) + ";\n" + END


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    html = PAGE.read_text()
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pat.search(html):
        sys.exit("markers not found in " + str(PAGE))
    new = pat.sub(lambda m: block(build()), html)
    if args.check:
        if new != html:
            sys.exit("explorable data block is stale: run src/export_explorable.py")
        print("explorable in sync")
        return
    PAGE.write_text(new)
    print(f"wrote data block into {PAGE.name} ({len(new) / 1024:.0f} KB page)")


if __name__ == "__main__":
    main()
