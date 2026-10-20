"""Write the explorable's data block from results/*.json and artifacts/lewm_planner.pt.

The explorable (course/explorables/31-world-model-from-pixels.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no measured number on the page is
typed by hand. The block carries the lewm seed-0 predictor with each
BatchNorm folded into the linear layer before it (float16, base64), the
encoder's output on a 48 x 48 grid of agent positions, and the probe, so the
page can plan with CEM for any agent and goal you drop.

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
import torch

import figures
import tworoom as W

PAGE = common.MODULE.parents[1] / "explorables" / "31-world-model-from-pixels.html"
BEGIN = "/* BEGIN DATA: written by modules/31-world-model-from-pixels/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def f16(a) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype=np.float32), dtype="<f2").tobytes()).decode()


def lin(w, b):
    return {"shape": list(w.shape), "w": f16(w), "b": f16(b)}


def fold(lin_w, lin_b, bn: dict, prefix: str):
    """A linear layer followed by an eval-mode BatchNorm, as one linear layer."""
    g, beta = bn[f"{prefix}.weight"].float(), bn[f"{prefix}.bias"].float()
    mu, var = bn[f"{prefix}.running_mean"].float(), bn[f"{prefix}.running_var"].float()
    s = g / torch.sqrt(var + 1e-5)
    return lin_w.float() * s[:, None], (lin_b.float() - mu) * s + beta


def weights() -> dict:
    ck = torch.load(common.ARTIFACTS / "lewm_planner.pt", weights_only=False)
    p = ck["predictor"]
    f = {k: v.float() for k, v in p.items()}
    w1, b1 = fold(f["proj.0.weight"], f["proj.0.bias"], p, "proj.1")
    return {"dim": ck["dim"],
            "act": [lin(f["act.0.weight"], f["act.0.bias"]), lin(f["act.2.weight"], f["act.2.bias"])],
            "net0": lin(f["net.0.weight"], f["net.0.bias"]), "ln": {"g": f16(f["net.1.weight"]), "b": f16(f["net.1.bias"])},
            "net3": lin(f["net.3.weight"], f["net.3.bias"]),
            "proj": [lin(w1, b1), lin(f["proj.3.weight"], f["proj.3.bias"])],
            "lookup": f16(ck["lookup"].float().numpy()), "lookup_n": int(ck["lookup"].shape[0]),
            "lookup_axis": [round(float(ck["lookup_axis"][0]), 3), round(float(ck["lookup_axis"][-1]), 3)],
            "probe": np.round(np.array(ck["probe"]), 6).tolist()}


def grid_episodes() -> dict:
    import grid

    arenas = grid.test_arenas(4, 3)
    out = {"arenas": [{"walls": a["walls"].astype(int).tolist(), "start": list(a["start"]), "goal": list(a["goal"]),
                       "bfs": a["bfs"]} for a in arenas], "episodes": {}}
    for f in sorted((common.RESULTS / "grid_runs").glob("*.json")):
        if f.name.startswith("followup_"):
            continue
        r = json.loads(f.read_text())
        out["episodes"][r["arm"]] = [{"success": e["success"], "steps": e["steps"], "path": e["path"]}
                                 for e in r["seeds"]["0"]["episodes"]]
    return out


def rnd(x):
    """Positions to whole library units (a 64-pixel frame has 3.5 units a pixel)."""
    return np.rint(np.asarray(x, dtype=float)).astype(int).tolist()


def build() -> dict:
    T, P, G, Lb, Wd = (common.read_results(n) for n in ("train", "plan", "grid", "library", "world"))
    rec = {}
    for name in ("lewm_s0__cem", "lewm_s0__shoot"):
        r = json.loads((common.RESULTS / "plan_runs" / f"{name}.json").read_text())
        rec[name] = {s: [{"success": e["success"], "steps": e["steps"], "path": rnd(e["path"]), "imagined": rnd(e["imagined"][:8]),
                          "first_rounds": rnd(e.get("first_rounds", []))} for e in r[s]["episodes"]] for s in ("cross", "same")}
    return {
        "world": {"img": W.IMG, "lo": W.LO, "hi": W.HI, "wall_x": W.WALL_X, "wall_half": W.WALL_HALF,
                  "wall_l": W.WALL_L, "wall_r": W.WALL_R, "door": [W.DOOR_Y - W.DOOR_HALF, W.DOOR_Y + W.DOOR_HALF],
                  "door_lo": W.DOOR_LO, "door_hi": W.DOOR_HI, "border": W.BORDER, "speed": W.SPEED, "reach": W.REACH},
        "data": {k: Wd["data"][k] for k in ("episodes", "transitions", "crossing_share")},
        "plan_settings": P["settings"],
        "tasks": {s: P["tasks"][s]["first"] for s in ("cross", "same")},
        "recorded": rec,
        "plan_table": figures.plan_table(P),
        "train": {a: {k: T["arms"][a][k] for k in ("probe_r2_xy", "erank", "spearman_geodesic", "dim")}
                  for a in T["arms"]},
        "random_weights_r2": T["random_weights"]["probe_r2_xy"],
        "grid": {"arms": {a: {k: G["arms"][a][k] for k in ("plan_success", "agent_r2", "goal_offset_r2", "erank")}
                          for a in G["arms"]}, "baselines": G["baselines"], **grid_episodes(),
                 "followup": {a: r["agent_probe_plan_success"]
                              for a, r in common.read_results("grid_followup")["arms"].items()}},
        "library": {"world": Lb["world"], "losses": Lb["losses"], "plan": Lb["plan"]},
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
