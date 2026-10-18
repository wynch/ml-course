"""Write the explorable's data block from results/*.json and artifacts/models.pt.

The explorable (course/explorables/25-world-models-and-planning.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no measured number on the page is
typed by hand. The block also carries the trained weights (float16, base64),
so the page can run both planners on an arena you draw.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys

import common
import numpy as np

PAGE = common.MODULE.parents[1] / "explorables" / "25-world-models-and-planning.html"
BEGIN = "/* BEGIN DATA: written by modules/25-world-models-and-planning/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def f16(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f2").tobytes()).decode()


def weights() -> dict:
    import torch

    ck = torch.load(common.ARTIFACTS / "models.pt", weights_only=False)
    oh = ck["state"]["onehot"]
    jp = ck["state"]["latent"]
    out = {"onehot": [], "target": [], "predictor": []}
    for i in (0, 2, 4):
        out["onehot"].append({"shape": list(oh[f"net.{i}.weight"].shape), "w": f16(oh[f"net.{i}.weight"]),
                              "b": f16(oh[f"net.{i}.bias"])})
    for part, key in (("target", "target.net"), ("predictor", "predictor.net")):
        for i in (0, 2):
            w = jp[f"{key}.{i}.weight"]
            out[part].append({"shape": list(w.shape), "w": f16(w), "b": f16(jp[f"{key}.{i}.bias"])})
    out["probe"] = np.round(ck["probes"]["latent"], 6).tolist()             # (33, 2): 32 weights, then the bias
    return out


def build() -> dict:
    W, R, P = common.read_results("world"), common.read_results("models"), common.read_results("plan")
    A = P["agents"]
    return {
        "world": {"goal_reachable_share": W["raw_arenas"]["goal_reachable_share"],
                  "wall_share": W["raw_arenas"]["wall_share_mean"], "blocked_share": W["train_data"]["blocked_share"]},
        "models": {"onehot_next_cell": R["onehot"]["next_cell_acc_held_out"],
                   "rollout": R["rollout_cell_acc"], "probe_r2": R["keeps"]["probe_r2"],
                   "jepa_loss": R["jepa"]["pred_loss_held_out"], "meters": R["jepa"]["meters"],
                   "meters_random": R["jepa_random"]["meters"], "params": R["params"]},
        "plan": {"shots": P["shots"], "horizon": P["horizon"], "budget": P["budget"], "n": P["arenas"]["n"],
                 "success": {a: A[a]["success"] for a in A},
                 "by_distance": {a: A[a]["by_distance"] for a in A},
                 "arenas": P["arenas"]["first"],
                 "episodes": {a: [{"success": e["success"], "steps": e["steps"], "path": e["path"]}
                                  for e in A[a]["episodes"]] for a in ("onehot", "latent", "random")}},
        "weights": weights(),
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
    b = block()
    new = pattern.sub(lambda _: b, page)
    if args.check:
        if new != page:
            sys.exit(f"{PAGE.name} is stale: run src/export_explorable.py")
        print(f"{PAGE.name} is in sync with results/ and artifacts/")
        return
    PAGE.write_text(new)
    print(f"wrote the data block of {PAGE.name} ({len(b) // 1024} KB)")


if __name__ == "__main__":
    main()
