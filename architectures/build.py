# /// script
# requires-python = ">=3.10"
# dependencies = ["jsonschema"]
# ///
"""Validate the architecture records and write them into the explorer page.

The page (course/explorables/05b-architecture-explorer.html) holds one
generated block between the BEGIN DATA and END DATA markers: every record that
passes validation, the glossary, the module titles the glossary links to, and
the planned models that have no record yet. No fetch() at run time, so the
offline reader sees everything.

The block also holds, per record, the parts parts.py derives from its fields;
the page's part filter and each definition's list of models read them.

A record that fails validation stops the build (exit 1, page untouched) unless
--lenient is given; then it is left out of the page and listed. --check never
writes, and fails when the page is stale or when any record fails.

    uv run course/architectures/build.py                # rewrite the block
    uv run course/architectures/build.py --lenient      # local work: leave failing records out
    uv run course/architectures/build.py --check        # exit 1 if a record fails or the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from validate import covered, leaves, load_glossary, record_problems  # noqa: E402  (the checks validate.py runs)
from parts import derived_parts  # noqa: E402

COURSE = HERE.parent
PAGE = COURSE / "explorables" / "05b-architecture-explorer.html"
BEGIN = "/* BEGIN DATA: written by architectures/build.py; do not edit */"
END = "/* END DATA */"
MODULE_TITLE = re.compile(r"^#\s*(?:Module|Lab)\s+([0-9]+[a-z½]?)\s*[—:-]\s*(.+?)\s*$")


def check_record(path: Path, rec: dict, validator, gloss: set[str], alias: dict, ids: set[str]) -> list[str]:
    """Return the problems with one record; an empty list means it passes."""
    problems = [f"schema: {list(e.path)} {e.message[:120]}" for e in validator.iter_errors(rec)]
    if problems:
        return problems
    if rec["id"] != path.stem:
        problems.append(f"id {rec['id']!r} does not match file name {path.name}")
    fs = rec["field_sources"]
    problems += [f"no source for {p}" for p in leaves(rec) if not covered(p, fs)]
    problems += [f"unknown ref {r} in {k}" for k, refs in fs.items() for r in refs if r not in rec["refs"]]
    problems += record_problems(rec, gloss, alias, ids)
    return problems


def modules() -> dict:
    """Module number -> folder and title, for the glossary's module links."""
    out = {}
    for readme in sorted(COURSE.glob("modules/*/README.md")):
        slug = readme.parent.name
        num = slug.split("-", 1)[0]
        first = readme.read_text().splitlines()[0]
        m = MODULE_TITLE.match(first)
        title = m.group(2).split(": ", 1)[0] if m else slug
        out[num] = {"slug": slug, "title": title}
    return out


def build() -> tuple[dict, list[tuple[str, list[str]]], list[str]]:
    schema = json.loads((HERE / "schema.json").read_text())
    glossary = json.loads((HERE / "glossary.json").read_text())
    planned = json.loads((HERE / "model_list.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    gloss, alias = load_glossary(HERE / "glossary.json")

    good, failed, loaded = [], [], []
    for path in sorted((HERE / "records").glob("*.json")):
        try:
            loaded.append((path, json.loads(path.read_text())))
        except json.JSONDecodeError as e:
            failed.append((path.stem, [f"not valid JSON: {e}"]))
    ids = {rec.get("id") for _, rec in loaded}
    for path, rec in loaded:
        problems = check_record(path, rec, validator, gloss, alias, ids)
        if problems:
            failed.append((path.stem, problems))
        else:
            good.append(rec)

    ids = {r["id"] for r in good}
    orphans = [f"{r['id']} -> {r['parent_id']}" for r in good if r.get("parent_id") and r["parent_id"] not in ids]
    good.sort(key=lambda r: (r["release_date"]["value"] or "9999", r["id"]))
    data = {
        "records": good,
        "parts": {r["id"]: sorted(derived_parts(r, gloss)) for r in good},
        "glossary": {"groups": glossary["groups"], "terms": glossary["terms"]},
        "modules": modules(),
        "planned": [
            {"id": m["id"], "name": m["name"], "org": m["org"], "date": m["date"], "open": m["open_weights"]}
            for m in planned if m["id"] not in ids
        ],
    }
    return data, failed, orphans


def block(data: dict) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    text = text.replace("</", "<\\/")  # never close the script tag from inside a string
    return f"{BEGIN}\nconst DATA = {text};\n{END}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 if a record fails or the page is stale")
    ap.add_argument("--lenient", action="store_true", help="leave failing records out instead of stopping")
    args = ap.parse_args()
    data, failed, orphans = build()
    page = PAGE.read_text()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(page):
        sys.exit(f"markers not found in {PAGE}")
    new_block = block(data)
    new = pattern.sub(lambda _: new_block, page)

    print(f"{len(data['records'])} records pass, {len(failed)} left out, {len(data['planned'])} planned without a record")
    for rid, problems in failed:
        print(f"  left out {rid}:")
        for p in problems[:8]:
            print(f"    {p}")
        if len(problems) > 8:
            print(f"    ... {len(problems) - 8} more")
    if orphans:
        print("  parent not among the records (lineage stops there): " + ", ".join(orphans))

    if failed and not args.lenient:
        sys.exit(f"{len(failed)} record(s) fail validation; fix them, or pass --lenient to leave them out"
                 + ("" if args.check else " (page not written)"))
    if args.check:
        if new != page:
            sys.exit(f"{PAGE.name} is stale: run architectures/build.py")
        print(f"{PAGE.name} is in sync with architectures/records/")
        return
    PAGE.write_text(new)
    print(f"wrote the data block of {PAGE.name} ({len(new_block) // 1024} KB)")


if __name__ == "__main__":
    main()
