"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/13-inside-vision-and-jepa.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no number on the page is typed by hand.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys

from PIL import Image

from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "13-inside-vision-and-jepa.html"
BEGIN = "/* BEGIN DATA: written by modules/13-inside-vision-and-jepa/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def shrink(b64: str, px: int = 72) -> str:
    """Re-encode a stored JPEG tile smaller for the page (the JSON keeps the larger one)."""
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((px, px), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def build() -> dict:
    v = read_results("v_neuron")
    va = read_results("v_neuron_acts")
    fv = read_results("feature_vis")
    clip = read_results("clip")
    jp = read_results("jepa_probe")
    return {
        "v": {
            "channels": v["channels"],
            "target": v["target"],
            "sets": v["sets"],
            "acts": va["trained"],
            "random55": va["random_55"],
            "gap": v["trained"]["gap_by_channel"],
            "gapTrucks": v["trained"]["gap_trucks_by_channel"],
            "rank": v["trained"]["rank_55"],
            "randomGap": v["random"]["gap_55"],
            "thumbs": {s: [{"a": t["a"], "j": t["jpg"]} for t in va["thumbs"][s]] for s in v["sets"]},
        },
        "renders": [{"c": r["channel"], "a": r["activation"], "best": r["best_real_image"], "j": r["jpg"]}
                    for r in fv["renders"]],
        "renderSteps": fv["steps"],
        "clip": {
            "trained": clip["probe"]["accuracy_trained"],
            "random": clip["probe"]["accuracy_random_weights"],
            "nTest": clip["n_test"],
            "neurons": [{"id": n["id"], "max": n["max"], "median": n["median_all"],
                         "top": [{"c": t["class"], "a": t["activation"], "j": shrink(t["jpg"])} for t in n["top"]]}
                        for n in clip["neurons"]],
        },
        "jepa": {
            "probes": jp["probes"],
            "shuffled": jp["shuffled_label_r2"],
            "cap": jp["cap"],
            "cappedShare": jp["capped_share"],
            "nEpisodes": jp["n_episodes"],
            "nTest": jp["n_test"],
            "scatter": jp["scatter"],
            "hist": jp["label_histogram"],
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
