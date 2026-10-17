"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/12-features-and-dictionaries.html) holds
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

PAGE = MODULE.parents[1] / "explorables" / "12-features-and-dictionaries.html"
BEGIN = "/* BEGIN DATA: written by modules/12-features-and-dictionaries/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    toy = read_results("toy")
    sae = read_results("sae")
    g2 = read_results("gpt2_sae")
    st = read_results("steer")
    main = next(r for r in sae["sweep"] if r["k"] == sae["k_main"])

    def char_unit(u, kind):
        return {"id": u["id"], "kind": kind, "index": u["index"], "max": u["max"], "freq": u["freq"],
                "contexts": [{"t": c["text"], "f": c["fire"], "a": c["activation"], "s": c["share_of_max"]}
                             for c in u["contexts"]]}

    return {
        "toy": {
            "sparsities": toy["sparsities"],
            "A": {"n": toy["setups"]["A"]["n"], "m": toy["setups"]["A"]["m"],
                  "frames": [{"norms": f["norms"], "itf": f["interference"], "gram": f["gram"],
                              "kept": f["n_represented"], "loss": f["loss"]} for f in toy["setups"]["A"]["frames"]]},
            "B": {"n": toy["setups"]["B"]["n"], "m": toy["setups"]["B"]["m"],
                  "frames": [{"W": f["W"], "kept": f["n_represented"]} for f in toy["setups"]["B"]["frames"]]},
        },
        "sae": {
            "k": sae["k_main"], "n_latents": 2048, "fve": main["fve"], "l0": main["l0"], "dead": main["dead"],
            "alive": main["alive_0.1pct"], "loss": sae["lm_loss"], "loss_sae": main["lm_loss_spliced"],
            "recovered": main["loss_recovered_vs_mean_ablation"],
            "purity": {k: sae["purity"][k] for k in ("latents_median", "neurons_median",
                                                      "latents_share_ge_0.9", "neurons_share_ge_0.9")},
            "units": [char_unit(u, "latent") for u in sae["latents"]] + [char_unit(u, "neuron") for u in sae["neurons"]],
        },
        "gpt2": {
            "stats": g2["stats"],
            "prompts": [{"prompt": p["prompt"], "tokens": p["tokens"],
                         "top": [{"i": d["index"], "a": d["activation"], "acts": d["acts_by_token"]} for d in p["top"]]}
                        for p in g2["prompts"]],
            "features": [{"i": f["index"], "url": f["url"], "freq": f["freq"], "max": f["corpus_max"],
                          "contexts": [{"t": c["tokens"], "f": c["fire"], "a": c["activation"], "s": c["share_of_max"]}
                                       for c in f["contexts"]]} for f in g2["features"]],
        },
        "steer": {
            "layer": st["layer"], "requests": st["requests"], "system": st["system_prompt"],
            "conditions": [{"name": n, "mean": c["mean_p_positive"], "p": c["p_positive"], "replies": c["replies"]}
                           for n, c in st["conditions"].items()],
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
