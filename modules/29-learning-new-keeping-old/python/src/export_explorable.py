"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/29-learning-new-keeping-old.html) holds one
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

PAGE = common.MODULE.parents[1] / "explorables" / "29-learning-new-keeping-old.html"
BEGIN = "/* BEGIN DATA: written by modules/29-learning-new-keeping-old/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"
ARMS = ["full", "full_slow", "lora4", "lora32", "replay", "distill"]
SEEDS = (0, 1, 2)


def run(name: str) -> dict:
    return json.loads((common.RESULTS / "runs" / f"{name}.json").read_text())


def build() -> dict:
    F = common.read_results("forget")
    S = common.read_results("start")
    out = {"steps": F["steps"], "start": {"old_loss": S["final"]["old_loss"], "new_loss": S["final"]["new_loss"],
                                          "samples": S["samples"]},
           "one_number": F["one_number"], "matched_new_loss": F["matched"]["new_loss"], "runs": {}, "arms": {}}
    for a in ARMS:
        out["arms"][a] = {"samples": F["arms"][a]["samples_s0"], "spec": F["arms"][a]["spec"],
                          "trainable": F["arms"][a]["trainable_parameters"]}
        for s in SEEDS:
            lg = run(f"{a}_s{s}")["log"]
            out["runs"][f"{a}_s{s}"] = {k: [round(r[k], 4) for r in lg] for k in ("old_loss", "new_loss", "kl_new", "kl_old")}
    P = common.read_results("plasticity")
    out["plasticity"] = {a: {"acc": v["online_acc_mean_curve"], "dead": v["health"]["dead"],
                             "erank": v["health"]["erank"]} for a, v in P["arms"].items()}
    if (common.RESULTS / "library.json").exists():
        L = common.read_results("library")
        out["library"] = {a: v["log"] for a, v in L["arms"].items()}
        out["library_rho"] = L["one_number"]["spearman_kl_new"]
        M = common.read_results("library_memory")
        out["memory"] = {"weights": M["base_weight_bytes"], "lora": M["lora_r8_weight_bytes"],
                         "grads": M["runs"]["plain"]["lora_grad_bytes"], "adam": M["runs"]["plain"]["adam_state_bytes"],
                         "saved": M["runs"]["plain"]["saved_for_backward_bytes"],
                         "saved_ckpt": M["runs"]["checkpointing"]["saved_for_backward_bytes"],
                         "params": M["base_parameters"], "batch": M["batch"], "tokens": M["tokens_per_sequence"]}
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
