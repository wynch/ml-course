"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/33-text-by-denoising.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no measured number on the page is
typed by hand. Nothing on the page runs a model: it replays recorded
samples and draws the measured curves.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import common

PAGE = common.MODULE.parents[1] / "explorables" / "33-text-by-denoising.html"
BEGIN = "/* BEGIN DATA: written by modules/33-text-by-denoising/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    F, P = common.read_results("flow2d"), common.read_results("flow2d_points")
    S, T = common.read_results("sample"), common.read_results("traces")
    B, E, R = common.read_results("block"), common.read_results("evaluate"), common.read_results("repeat")
    pts = {k: [[round(x, 2), round(y, 2)] for x, y in v[:400]] for k, v in P.items()
           if not k.endswith("_paths") and (k == "held_out" or k.split("_")[0] in ("fm", "ddpm"))}
    return {
        "flow": {"steps": F["steps"], "results": F["results"], "floor": F["eval"]["floor_two_spiral_draws"], "points": pts},
        "traces": T["traces"],
        "sample": {"steps": S["steps"],
                   "results": {m: {r: {n: {k: v[k] for k in ("word_score", "judge_nll", "distinct4")} for n, v in rows.items()}
                                   for r, rows in byrule.items()} for m, byrule in S["results"].items()},
                   "references": {k: {m: v[m] for m in ("word_score", "judge_nll", "distinct4")}
                                  for k, v in S["references"].items()}},
        "block": {"traces": B["traces"],
                  "table": {tot: {k: {m: v[m] for m in ("word_score", "seconds_for_64", "calls", "positions")}
                                  for k, v in rows.items()} for tot, rows in B["results"].items()}},
        "likelihood": E["likelihood_nats_per_char"],
        "infill": E["infill"]["results"],
        "repeat": {size: repeat_size(S_) for size, S_ in R["sizes"].items()},
    }


def repeat_size(S: dict) -> dict:
    """Every arm on one epoch axis (the longest one); a missing mark is null."""
    names = [n for n in ("ar", "ard3", "mdm", "mdmlong") if n in S]
    axis = max((S[n]["epochs"] for n in names), key=len)
    out = {"epochs": axis, "unique_chars": S["unique_chars"]}
    for n in names:
        a = S[n]
        keys = ("val_mean", "train_mean") + (("val_l2r_mean",) if n.startswith("mdm") else ())
        out[n] = {k: [dict(zip(a["epochs"], a[k])).get(e) for e in axis] for k in keys}
    return out


def render(page: str, data: dict) -> str:
    block = BEGIN + "\nconst DATA = " + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n" + END
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    assert pat.search(page), "markers not found"
    return pat.sub(lambda _: block, page)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    page = PAGE.read_text()
    new = render(page, build())
    if a.check:
        if new != page:
            print("explorable is stale: run src/export_explorable.py")
            sys.exit(1)
        print("explorable in sync")
        return
    PAGE.write_text(new)
    print(f"  wrote {PAGE} ({len(new) / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
