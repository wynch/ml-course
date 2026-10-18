"""Write the explorable's data block from results/*.json and artifacts/line_net.pt.

The explorable (course/explorables/20-in-context-learning.html) holds one generated
block between the BEGIN DATA and END DATA markers. This script rewrites that block
and nothing else, so no number on the page is typed by hand. The line net's weights
go in as base64 float32, so the page runs the trained net itself.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys

from common import ARTIFACTS, MODULE, read_results

import torch

PAGE = MODULE.parents[1] / "explorables" / "20-in-context-learning.html"
BEGIN = "/* BEGIN DATA: written by modules/20-in-context-learning/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def r4(xs):
    return [round(float(x), 5) for x in xs]


def weights() -> dict:
    sd = torch.load(ARTIFACTS / "line_net.pt")
    out = {}
    for k, v in sd.items():
        a = v.detach().cpu().numpy().astype("<f4")
        out[k] = {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}
    return out


def build() -> dict:
    icl = read_results("icl")
    gd = read_results("gd_demo")
    fs = read_results("fewshot")
    line = read_results("line")
    m = icl["mse_over_d"]
    return {
        "icl": {"n": m["n"], **{k: r4(m[k]) for k in ("net", "least_squares", "ridge_0.1", "knn_3", "one_gd_step",
                                                    "control_net_on_shuffled", "main_net_on_shuffled")},
                "bars": icl["bars"], "order": icl["order_permutation"], "steps": icl["config"]["steps"],
                "params": icl["config"]["params"]},
        "gd": {"n": gd["implied"]["n"], "cos_net_ls": r4(gd["implied"]["cos_net_ls"]),
               "cos_net_gd": r4(gd["implied"]["cos_net_gd"]), "cos_gd_ls": r4(gd["implied"]["cos_gd_ls"]),
               "probe_n": gd["config"]["probe_n"], "layers": gd["config"]["layers"],
               "probes": {net: {t: r4(v) for t, v in d.items()} for net, d in gd["probes_r2"].items()}},
        "fewshot": {"conditions": {k: {"acc": round(v["accuracy"], 4), "ci": r4(v["ci95"]),
                                       "pos": round(v["share_positive"], 4), "tokens": round(v["mean_prompt_tokens"], 1)}
                                   for k, v in fs["conditions"].items()},
                    "n": fs["config"]["n_test"], "majority": round(fs["config"]["majority_share"], 4),
                    "header": fs["config"]["header"], "examples": fs["examples"]},
        "line": {"shape": {k: line["config"][k] for k in ("d_model", "heads", "d_mlp", "layers", "params", "steps")},
                 "mse": {"n": line["mse_over_d"]["n"], "net": r4(line["mse_over_d"]["net"]),
                         "ridge": r4(line["mse_over_d"]["ridge_0.1"])},
                 "demo": {k: (r4(v) if isinstance(v, list) else v) for k, v in line["demo"].items()},
                 "weights": weights()},
    }


def block() -> str:
    data = json.dumps(build(), ensure_ascii=False, separators=(",", ":"))
    return f"{BEGIN}\nconst DATA = {data};\n{END}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    page = PAGE.read_text()
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pat.search(page):
        sys.exit(f"markers not found in {PAGE}")
    new = pat.sub(lambda _: block(), page, count=1)
    if args.check:
        if new != page:
            print("explorable data block is stale; run src/export_explorable.py")
            sys.exit(1)
        print("explorable data block is up to date")
        return
    PAGE.write_text(new)
    print(f"  wrote {PAGE.relative_to(MODULE.parents[1])}")


if __name__ == "__main__":
    main()
