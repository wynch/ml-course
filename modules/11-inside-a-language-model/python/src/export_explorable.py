"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/11-inside-a-language-model.html) holds
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

PAGE = MODULE.parents[1] / "explorables" / "11-inside-a-language-model.html"
BEGIN = "/* BEGIN DATA: written by modules/11-inside-a-language-model/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    hooks = read_results("hooks")
    probe = read_results("probe")
    patch = read_results("patching")
    poles = read_results("poles")
    return {
        "hook": {
            "block": hooks["neuron"]["block"],
            "neuron": hooks["neuron"]["neuron"],
            "r": hooks["neuron"]["r"],
            "passage": hooks["passage"],
            "acts": hooks["passage_activations"],
            "loss_after_space": [hooks["edit"]["loss_after_space_before"],
                                 hooks["edit"]["loss_after_space_after"],
                                 hooks["edit"]["loss_after_space_top10_zeroed"]],
        },
        "probe": {
            "trained": probe["accuracy_trained"],
            "random": probe["accuracy_random_weights"],
            "majority": probe["majority_baseline"],
            "plateau": probe["plateau_layer"],
            "best": probe["best_layer"],
            "n_test": 872,
        },
        "patch": {
            "clean": patch["clean"],
            "corrupt": patch["corrupt"],
            "tokens_clean": patch["tokens_clean"],
            "tokens_corrupt": patch["tokens_corrupt"],
            "ld_clean": patch["ld_clean"],
            "ld_corrupt": patch["ld_corrupt"],
            "score": patch["score"],
            "best_subject": patch["best_subject_cell"],
            "last_first": patch["last_position_first_restoring_row"],
        },
        "poles": {
            "corpus": poles["corpus"],
            "rule": poles["rule"],
            "neurons": [
                {
                    "id": n["id"],
                    "layer": n["layer"],
                    "neuron": n["neuron"],
                    **{pole: {"max": n[pole]["max"],
                              "contexts": [{"t": c["tokens"], "f": c["fire"], "a": c["activation"],
                                            "s": c["share_of_max"], "r": c["regime"]}
                                           for c in n[pole]["contexts"]]}
                       for pole in ("pos", "neg")},
                }
                for n in poles["neurons"]
            ],
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
