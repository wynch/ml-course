"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/21-retrieval-and-memory.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no number on the page is typed by hand.
The page ships only the 200 SQuAD paragraphs of section 3's sample (CC BY-SA 4.0).

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "21-retrieval-and-memory.html"
BEGIN = "/* BEGIN DATA: written by modules/21-retrieval-and-memory/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    emb = read_results("embed")
    ret = read_results("retrieval")
    rag = read_results("rag")
    ex = ret["explorable"]
    gens = json.loads((MODULE / "results" / rag["generation_file"]).read_text())
    gold = gens["conditions"]["gold"]["paragraph"]
    reader = []
    for i in range(len(ex["questions"])):
        row = {}
        for c in rag["conditions"]:
            pq = rag["per_question"][c]
            row[c] = {"p": pq["pred"][i], "em": pq["em"][i], "f1": pq["f1"][i]}
            if c in ("bm25", "minilm"):
                row[c]["hit"] = gens["conditions"][c]["paragraph"][i] == gold[i]
        reader.append(row)
    return {
        "bm25": {"k1": ret["bm25"]["k1"], "b": ret["bm25"]["b"]},
        "questions": ex["questions"], "answers": ex["answers"], "titles": ex["titles"], "passages": ex["passages"],
        "cosine_top": ex["cosine_top"], "py_bm25_top": ex["bm25_top"], "full_rank": ex["full_rank"],
        "reader": reader,
        "recall": ret["recall"], "mrr": ret["mrr"], "curves": ret["curves"],
        "corpus": {"paragraphs": ret["data"]["n_paragraphs"], "questions": ret["data"]["n_questions"],
                   "articles": ret["data"]["n_articles"]},
        "rag": {"n": rag["n"], "conditions": rag["conditions"], "table": rag["table"]},
        "embed": {n: {"gold": m["cos_question_gold"]["median"], "random": m["cos_question_random"]["median"],
                      "beats": m["gold_beats_random"], "nn_same": m["nn_same_article"]}
                  for n, m in emb["models"].items()},
        "chance_same_article": emb["chance_same_article"],
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
