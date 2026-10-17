"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/17-learning-from-preferences.html) holds
one generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no number on the page is typed by hand.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "17-learning-from-preferences.html"
BEGIN = "/* BEGIN DATA: written by modules/17-learning-from-preferences/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"
N_SAMPLES = 6


def build() -> dict:
    bt = read_results("bt_synthetic")
    rh = read_results("reward_head")
    hand = read_results("dpo_by_hand")
    lb = read_results("length_bias")

    def dpo(cfg):
        d = read_results(f"dpo_{cfg}")
        h, g = d["heldout"], d["generations"]
        return {"steps": d["settings"]["steps"], "pairs": h["pairs"],
                "acc_after": h["implicit_reward_accuracy_after"]["strict"],
                "ref_acc": h["reference_logp_accuracy"]["strict"],
                "ref_acc_tok": h["reference_logp_per_token_accuracy"]["strict"],
                "minutes": d["guard"]["minutes"], "rss": d["guard"]["peak_rss_gb"],
                "mps": d["guard"]["peak_mps_driver_gb"],
                "gen": {k: g[k] for k in ("before", "after", "prompts", "max_new_tokens",
                                          "head_win_rate_after_vs_before")},
                "log": [{"s": x["step"], "loss": x["loss"], "rc": x["rewards/chosen"], "rr": x["rewards/rejected"]}
                        for x in d["log_history"]],
                "samples": g["samples"][:N_SAMPLES]}

    return {
        "bt": {"s_true": bt["s_true"], "main": {k: bt["main"][k] for k in ("comparisons", "s_fit", "kendall_tau")},
               "sweep": bt["sweep"]},
        "head": {"acc": rh["heads"]["last_token"]["heldout_accuracy"],
                 "acc_mean": rh["heads"]["reply_mean"]["heldout_accuracy"],
                 "longer": rh["baselines"]["longer_reply_wins"], "pairs": rh["data"]["heldout_pairs"]},
        "pair": {k: hand["per_pair"][0][k] for k in ("pi_chosen", "pi_rejected", "ref_chosen", "ref_rejected")},
        "check": {"hand": hand["hand"]["loss"], "trl": hand["trl"]["loss"]},
        "dpo": {"preview": dpo("preview"), "full": dpo("full")},
        "length": {"raw_longer": lb["data_raw"]["chosen_longer"], "raw_pairs": lb["data_raw"]["pairs"],
                   "length_only": lb["length_only_reward"]["heldout_accuracy"],
                   "head": lb["reward_head"], "dpo": lb["dpo"].get("full")},
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
