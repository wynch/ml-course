"""Write the explorable's data block from results/.

The explorable (course/explorables/22-reasoning-at-test-time.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no number on the page is typed by hand.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import sys

from common import GEN, MODULE, read_jsonl, read_results

PAGE = MODULE.parents[1] / "explorables" / "22-reasoning-at-test-time.html"
BEGIN = "/* BEGIN DATA: written by modules/22-reasoning-at-test-time/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    a = read_results("analysis")
    probs = {p["id"]: p for p in read_jsonl(GEN / "problems.jsonl")}
    greedy = {r["id"]: r for r in read_jsonl(GEN / "greedy.jsonl")}
    samples = {r["id"]: r["samples"] for r in read_jsonl(GEN / "samples.jsonl")}
    s1, s2, s3 = a["section1"], a["section2"], a["section3"]
    rows = []
    for p in a["per_problem"]:
        i = p["id"]
        rows.append({
            "q": probs[i]["question"], "g": p["gold"], "d": p["direct"],
            "gt": greedy[i]["text"], "ga": p["greedy"], "gn": greedy[i]["tokens"],
            "s": [{"t": s["text"], "a": ans, "n": s["tokens"], "v": sc, "l": lo}
                  for s, ans, sc, lo in zip(samples[i], p["samples"], p["scores"], p["logodds"])],
        })
    return {
        "model": a["model"], "weights": a["weights"], "n": a["n_problems"], "temp": a["temperature"],
        "direct_acc": s1["direct_acc"], "greedy_acc": s1["greedy_acc"],
        "direct_tokens": s1["direct_mean_tokens"], "greedy_tokens": s1["greedy_mean_tokens"],
        "vote5": s2["vote5_acc"], "auc": s3["auc"],
        "problems": rows,
    }


def block() -> str:
    data = json.dumps(build(), ensure_ascii=False, separators=(",", ":"))
    return f"{BEGIN}\nconst DATA = {data};\n{END}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    page = PAGE.read_text()
    start, stop = page.index(BEGIN), page.index(END) + len(END)
    new = page[:start] + block() + page[stop:]
    if args.check:
        if new != page:
            print("explorable is stale: run src/export_explorable.py")
            sys.exit(1)
        print("explorable in sync")
        return
    PAGE.write_text(new)
    print(f"wrote {PAGE.name} ({len(new) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
