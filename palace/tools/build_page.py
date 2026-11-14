#!/usr/bin/env python3
"""Write the palace's data into ../index.html, between the BEGIN DATA and END DATA markers.

The block holds source/palace.json, source/cards.json (card text for objects with no
move) and the text of every move the palace uses, read from the One Hundred Moves
page (course/moves/index.html): titles and bodies from its OLD table and its JSON
block, deeper layers from the JSON block. Nothing is fetched at run time.

    python3 course/palace/tools/build_page.py           # write the block
    python3 course/palace/tools/build_page.py --check   # write nothing; exit 1 if the page is stale
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAGE = ROOT / "index.html"
MOVES = ROOT.parent / "moves" / "index.html"
BEGIN = "<!-- BEGIN DATA: written by palace/tools/build_page.py; do not edit -->"
END = "<!-- END DATA -->"

JS_STR = r"'((?:[^'\\]|\\.)*)'"


def js_unescape(s):
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), s)


def moves_text():
    """Every move's title, body and deeper layer, as the Moves page shows them."""
    html = MOVES.read_text()
    m = re.search(r'<script type="application/json" id="moves-text">(.*?)</script>', html, re.S)
    if not m:
        sys.exit(f"moves-text block not found in {MOVES}")
    block = json.loads(m.group(1))
    script = html[m.end():]
    old = script[script.index("const OLD=["):]
    old = old[:old.index("\n];")]
    rows = re.findall(r"^\[F\.\w+\((?:[^()]|\([^()]*\))*\)," + JS_STR + "," + JS_STR + r"\],?$", old, re.M)
    if len(rows) != 100:
        sys.exit(f"expected 100 OLD rows in {MOVES}, found {len(rows)}")
    out = {}
    for k, (title, body) in enumerate(rows, 1):
        out[f"a{k}"] = {"title": js_unescape(title), "body": js_unescape(body)}
    for mid, t in block["moves"].items():
        e = out.setdefault(mid, {})
        for key in ("title", "body", "deeper"):
            if key in t:
                e[key] = t[key]
    return out


def build():
    palace = json.loads((ROOT / "source" / "palace.json").read_text())
    cards = json.loads((ROOT / "source" / "cards.json").read_text())
    allm = moves_text()
    used = {mv for r in palace["rooms"] for o in r["objects"] for mv in o["moves"]}
    used |= set(palace["prologue"]["moves"]) | set(palace["epilogue"]["moves"])
    missing = sorted(used - set(allm))
    if missing:
        sys.exit(f"moves not found on the Moves page: {missing}")
    known = {o["id"] for r in palace["rooms"] for o in r["objects"]}
    stray = sorted(set(cards) - known - {"_note"})
    if stray:
        sys.exit(f"cards.json names unknown objects: {stray}")
    data = {"palace": palace, "moves": {k: allm[k] for k in sorted(used, key=lambda x: (x[0], int(x[1:])))},
            "cards": {k: v for k, v in cards.items() if k != "_note"}}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    text = text.replace("</", "<\\/")
    return f'{BEGIN}\n<script type="application/json" id="palace-data">{text}</script>\n{END}'


def main():
    check = "--check" in sys.argv[1:]
    page = PAGE.read_text()
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pat.search(page):
        sys.exit(f"markers not found in {PAGE}")
    new = pat.sub(lambda _: build(), page)
    if check:
        if new != page:
            print(f"{PAGE.name} is stale: run tools/build_page.py")
            sys.exit(1)
        print("palace page is current")
        return
    PAGE.write_text(new)
    print(f"wrote data block into {PAGE}")


if __name__ == "__main__":
    main()
