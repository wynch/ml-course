"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/15-geometry-of-representations.html) holds
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

PAGE = MODULE.parents[1] / "explorables" / "15-geometry-of-representations.html"
BEGIN = "/* BEGIN DATA: written by modules/15-geometry-of-representations/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    rg = read_results("regions")
    idr = read_results("intrinsic_dim")
    mf = read_results("manifold")
    ck = read_results("cka")
    toy = rg["toy"]
    pick = lambda e: {"twonn": e["twonn"], "mle": e["mle"]}
    return {
        "toy": {"width": toy["width"], "box": toy["box"], "init": toy["weights_init"], "trained": toy["weights_trained"],
                "regions_init": toy["regions_init"], "regions_trained": toy["regions_trained"],
                "points": toy["points"], "labels": toy["labels"], "train_acc": toy["train_acc"]},
        "width": {"rows": [{k: w[k] for k in ("width", "neurons", "init_mean", "trained_mean", "init_layer1_mean",
                                                "trained_layer1_mean", "test_acc")} for w in rg["width"]["widths"]],
                  "ratios_trained": rg["width"]["ratio_per_doubling_trained"],
                  "ratios_init": rg["width"]["ratio_per_doubling_init"], "lines": rg["width"]["lines"]},
        "id": {
            "known": [{"d": r["true_dim"], **pick(r)} for r in idr["known"]],
            "known_n": idr["known"][0]["n"],
            "fashion": {k: pick(idr["fashion"][k]) for k in ("pixels", "hidden", "logits", "hidden_init", "logits_init")},
            "mlp_acc": idr["fashion"]["mlp_test_acc"],
            "vit_layers": idr["vit"]["per_layer_cls"], "vit_cls": pick(idr["vit"]["cls"]),
            "vit_pixels": pick(idr["vit"]["pixels_64"]), "vit_n": idr["vit"]["n_images"],
        },
        "pca": {k: v["components"] for k, v in mf["pca"].items()},
        "midpoints": {k: mf["midpoints"][k] for k in ("real_test_to_train", "same_class", "different_class")},
        "cka": {"pairs": ck["pairs"], "pixels_vs_hidden": ck["pixels_vs_hidden"], "n": ck["n_inputs"]},
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
