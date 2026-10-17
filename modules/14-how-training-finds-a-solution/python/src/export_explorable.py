"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/14-how-training-finds-a-solution.html)
holds one generated block between the BEGIN DATA and END DATA markers. This
script rewrites that block and nothing else, so no number on the page is typed
by hand.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "14-how-training-finds-a-solution.html"
BEGIN = "/* BEGIN DATA: written by modules/14-how-training-finds-a-solution/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    toy = read_results("optim_toy")
    mlp = read_results("optim_mlp")
    land = read_results("landscape")
    hf = read_results("hf_way")

    def grok(name):
        g = read_results(name)
        return {"wd": g["weight_decay"], "train100": g["train_hits_100"], "test95": g["test_crosses_95"],
                "final": g["final"], "n_train": g["n_train"], "n_test": g["n_test"],
                "curve": [[c["step"], round(c["train_acc"], 4), round(c["test_acc"], 4),
                           round(c["train_loss"], 4), round(c["test_loss"], 4), c["weight_norm"]] for c in g["curve"]]}

    return {
        "toy": {
            "target": toy["target"], "cap": toy["cap"], "momentum": toy["momentum"], "adam": toy["adam"],
            "kappa": toy["bowl"]["kappa"], "angle_deg": toy["bowl"]["angle_deg"],
            "losses": {name: {"start": L["start"], "minimum": L["minimum"], "box": L["box"],
                              "best": {o: {"lr": v["best_lr"], "steps": v["best_steps"], "reach": v["lrs_that_reach"]}
                                       for o, v in L["optimisers"].items()}}
                       for name, L in toy["losses"].items()},
            "n_lrs": len(toy["lrs"]),
        },
        "mlp": {
            "lrs": mlp["lrs"], "epochs": mlp["epochs"], "steps": mlp["steps"], "batch": mlp["batch"],
            "runs": [{"o": r["optimiser"], "s": r["schedule"], "lr": r["lr"], "acc": r["test_acc"],
                      "curve": [c["test_acc"] for c in r["curve"]]} for r in mlp["runs"] if r["seed"] == 0],
            "best": mlp["best"],
            "hf": [{"o": r["optimiser"], "optim": r["optim"], "args": r["optim_args"], "lr": r["lr"], "acc": r["test_acc"]}
                   for r in hf["runs"]],
        },
        "land": {
            "line": land["line"], "limit": land["stability_limit"], "init_eig": land["init"]["top_eig"],
            "lr": land["lr"], "target": land["target_train_loss"],
            "nets": {k: {f: n[f] for f in ("batch", "epochs", "steps", "train_loss", "test_acc", "top_eig",
                                           "weight_norm", "line_train", "line_test")}
                     for k, n in land["nets"].items()},
        },
        "grok": {"wd1": grok("grok_wd1"), "wd0": grok("grok_wd0")},
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
