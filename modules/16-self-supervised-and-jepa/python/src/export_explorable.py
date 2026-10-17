"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/16-self-supervised-and-jepa.html) holds
one generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no measured number on the page is
typed by hand. (Panel 1's toy trains in the browser and needs no data.)

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "16-self-supervised-and-jepa.html"
BEGIN = "/* BEGIN DATA: written by modules/16-self-supervised-and-jepa/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"
VARIANTS = ("no-stop-grad", "stop-grad", "ema", "vicreg-var", "sigreg")
ROWS = (("random", "random encoder (floor)", "none"),
        ("mae", "masked reconstruction", "rebuild hidden 4×4 patches"),
        ("jepa", "I-JEPA-lite", "predict an EMA encoder's latents of hidden cells"),
        ("simclr", "SimCLR", "tell two augmented views of one image from the rest"),
        ("supervised", "supervised (ceiling)", "the labels"))


def build() -> dict:
    C = [read_results(f"collapse_{v}") for v in VARIANTS]
    P = {k: read_results(f"pretrain_{k}") for k, _, _ in ROWS}
    d = read_results("dinov2")
    pk = ("class_acc", "colour_r2", "rotation_acc")
    return {
        "collapse": {
            "budget": C[0]["budget_seconds"], "latent_dim": C[0]["latent_dim"], "meter_images": C[0]["meter_images"],
            "floor_10k": P["random"]["probes"]["class_acc_10k"],
            "runs": [{"variant": r["variant"], "steps": r["steps"], "class_acc_10k": r["class_acc_10k"], "final": r["final"],
                      "log": [{"seconds": p["seconds"], "std_norm": p["std_norm"], "erank": p["erank"],
                               "pred_loss": p["pred_loss"]} for p in r["log"]]} for r in C],
        },
        "keeps": {
            "encoder_params": P["simclr"]["encoder_params"], "budget": P["simclr"]["budget_seconds"],
            "rows": [{"key": k, "name": n, "what": w, "steps": P[k]["steps"], "p": {q: P[k]["probes"][q] for q in pk},
                      "class_acc_10k": P[k]["probes"]["class_acc_10k"]} for k, n, w in ROWS],
            "dinov2": {"what": "self-distillation and masked prediction (DINO + iBOT losses), 142M images",
                       "p": {q: d[q] for q in pk}, "train_images": d["train_images"],
                       "rot_images": d["rotation_images"]["train"]},
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
