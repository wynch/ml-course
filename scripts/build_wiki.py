#!/usr/bin/env python3
"""Render the private wiki to a static, mobile-first reader under wiki/.

Run from the private project, where the wiki sits next to course/:

    python3 course/scripts/build_wiki.py                 # the published set
    python3 course/scripts/build_wiki.py --types all     # every page
    python3 course/scripts/build_wiki.py --types concept,guide --pages overview

The published set leaves out Vincent's own work: the types in PRIVATE_TYPES
and the pages in PRIVATE_PAGES. --types or --pages replace that default.

Each page is one self-contained HTML file (inline CSS, no fonts, no scripts
except the search on the index), so it keeps working offline once opened.
Only rendered wiki text is published: raw sources stay private and are named,
never linked, and local paths are shown relative to the project root.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import posixpath
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SOURCE = PROJECT / "wiki"
OUT = ROOT / "wiki"
SKIP = {"log.md"}
PRIVATE_TYPES = {"experiment", "idea"}
# Analysis pages whose subject is the local projects, not papers.
PRIVATE_PAGES = {"analysis-experiments-second-chance", "analysis-local-project-acronyms"}
# Index labels such as "**16 experiment pages**", recounted after pruning.
COUNT_LABELS = {
    "source pages": "source", "source bundles": "source-bundle", "concept pages": "concept",
    "entity pages": "entity", "analysis pages": "analysis", "reading guides": "guide",
    "cross-cutting threads": "thread", "idea pages": "idea", "experiment pages": "experiment",
    "overview": "overview",
}

# Order matters: the project itself first, then its siblings, then any other
# home path. What is left must never contain a home directory or a vault link.
SCRUB = [
    (re.compile(r"obsidian://[^\s)\]>\"'`]*"), ""),
    (re.compile(r"file://"), ""),
    (re.compile(r"(?:/Users/[^/\s`\"'\])]+|~)/Perso/ml/"), ""),
    (re.compile(r"(?:/Users/[^/\s`\"'\])]+|~)/Perso/"), "../"),
    (re.compile(r"/Users/[^/\s`\"'\])]+/?"), "~/"),
    # Links written from inside wiki/ climb one level more than the project.
    (re.compile(r"(?<![\w./])\.\./\.\./(?=[\w-])"), "../"),
]
FORBIDDEN = re.compile(r"/Users/|~/Perso|obsidian://|file://")

CSS = """
:root{--paper:#f6f8f7;--panel:#fff;--ink:#1c2422;--muted:#5c6b67;--line:rgba(28,36,34,.14);
--accent:#1f918d;--soft:rgba(31,145,141,.10);--code:#eef2f0;
--body:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--paper:#121816;--panel:#1a2320;--ink:#e8eeec;
--muted:#9daba7;--line:rgba(232,238,236,.18);--accent:#37b3ae;--soft:rgba(55,179,174,.16);--code:#0d1211}}
*{box-sizing:border-box}
html{background:var(--paper);-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.6 var(--body);
overflow-wrap:break-word}
a{color:var(--accent)}
.bar{position:sticky;top:0;z-index:2;display:flex;gap:.5rem;align-items:center;
padding:.55rem 16px;background:var(--panel);border-bottom:1px solid var(--line);font-size:.9rem}
.bar a{text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:.2rem .7rem;color:var(--ink)}
.bar .sp{flex:1}
main{max-width:46rem;margin:0 auto;padding:1rem 16px 4rem}
h1{font-size:1.5rem;line-height:1.25;margin:.6rem 0 .5rem}
h2{font-size:1.2rem;margin:1.8rem 0 .5rem;line-height:1.3}
h3{font-size:1.05rem;margin:1.4rem 0 .4rem}
h4,h5,h6{font-size:1rem;margin:1.2rem 0 .3rem}
.meta{font-size:.82rem;color:var(--muted);margin:.2rem 0 1rem;display:flex;flex-wrap:wrap;gap:.3rem .5rem;align-items:center}
.meta .type{font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--accent)}
.tag{font-family:var(--mono);font-size:.74rem;background:var(--soft);border-radius:999px;padding:.05rem .5rem}
code{font-family:var(--mono);font-size:.86em;background:var(--code);border-radius:4px;padding:.05em .3em;overflow-wrap:anywhere}
pre{background:var(--code);border-radius:8px;padding:.8rem;overflow-x:auto;font-size:.82rem;line-height:1.5}
pre code{background:none;padding:0;overflow-wrap:normal}
.tw{overflow-x:auto;margin:1rem 0;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;font-size:.86rem;min-width:100%}
th,td{border-bottom:1px solid var(--line);padding:.4rem .55rem;text-align:left;vertical-align:top}
th{background:var(--soft)}
blockquote{margin:1rem 0;padding:.2rem .9rem;border-left:3px solid var(--accent);color:var(--muted)}
ul,ol{padding-left:1.3rem}
li{margin:.2rem 0}
hr{border:0;border-top:1px solid var(--line);margin:1.5rem 0}
.dead{color:var(--muted);border-bottom:1px dotted var(--muted)}
details.src{margin:2rem 0 0;border:1px solid var(--line);border-radius:8px;padding:.5rem .8rem;background:var(--panel)}
details.src summary{cursor:pointer;color:var(--muted);font-size:.9rem}
details.src li{font-size:.84rem}
.search{width:100%;font:inherit;font-size:16px;padding:.6rem .8rem;border:1px solid var(--line);
border-radius:10px;background:var(--panel);color:var(--ink);margin:.4rem 0 .3rem}
.hits{list-style:none;padding:0;margin:.5rem 0 1.5rem}
.hits li{padding:.45rem 0;border-bottom:1px solid var(--line)}
.hits small{color:var(--muted);display:block;font-size:.78rem}
.note{font-size:.84rem;color:var(--muted)}
"""

SEARCH_JS = """
(function(){
var pages=JSON.parse(document.getElementById('pages').textContent);
var box=document.getElementById('q'),hits=document.getElementById('hits'),body=document.getElementById('body');
function esc(s){return s.replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
function run(){
  var q=box.value.trim().toLowerCase();
  if(!q){hits.innerHTML='';body.hidden=false;return}
  var words=q.split(/\\s+/),out=[];
  pages.forEach(function(p){
    var hay=(p.title+' '+p.name+' '+p.type+' '+p.tags.join(' ')).toLowerCase();
    if(words.every(function(w){return hay.indexOf(w)>=0}))out.push(p)});
  body.hidden=true;
  hits.innerHTML=out.length?out.slice(0,200).map(function(p){
    return '<li><a href="'+esc(p.name)+'.html">'+esc(p.title)+'</a><small>'+esc(p.type)+
      (p.tags.length?' · '+esc(p.tags.join(', ')):'')+'</small></li>'}).join(''):
    '<li class="note">No page title or tag matches.</li>';
}
box.addEventListener('input',run);run();
})();
"""


def scrub(text: str) -> str:
    for pattern, replacement in SCRUB:
        text = pattern.sub(replacement, text)
    return text


def parse_list(value: str) -> list[str]:
    """Read an inline YAML list such as [a, "b, c", 'd'] without PyYAML."""
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return [value] if value else []
    items, current, quote = [], "", ""
    for char in value[1:-1]:
        if quote:
            if char == quote:
                quote = ""
            else:
                current += char
        elif char in "\"'" and not current.strip():
            quote = char
        elif char == ",":
            items.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        items.append(current.strip())
    return [item for item in items if item]


def split_front_matter(text: str) -> tuple[dict, str]:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict = {}
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            return meta, "\n".join(lines[index + 1:])
        key, _, value = line.partition(":")
        value = value.strip()
        if key in ("sources", "tags"):
            meta[key] = parse_list(value)
        else:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            meta[key.strip()] = value
    return {}, text


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_]+", "-", text).strip("-")


class Inline:
    """Inline Markdown for one page; knows which pages are published."""

    def __init__(self, published: set[str]):
        self.published = published

    def wiki_link(self, target: str, label: str | None) -> str:
        name, _, heading = target.partition("#")
        name = name.strip()
        if name.endswith(".md"):
            name = name[:-3]
        text = escape(label if label is not None else (name or heading))
        if name in self.published:
            anchor = f"#{slugify(heading)}" if heading else ""
            return f'<a href="{escape(name)}.html{anchor}">{text}</a>'
        if not name and heading:
            return f'<a href="#{slugify(heading)}">{text}</a>'
        return f'<span class="dead" title="not published">{text}</span>'

    def md_link(self, label_html: str, url: str) -> str:
        url = url.strip()
        if url.startswith("<") and url.endswith(">"):
            url = url[1:-1]
        url = url.split(" ")[0] if " \"" in url else url
        if re.match(r"(https?:|mailto:)", url):
            return f'<a href="{escape(url)}" rel="noreferrer">{label_html}</a>'
        if url.startswith("#"):
            return f'<a href="{escape(url)}">{label_html}</a>'
        path, _, anchor = url.partition("#")
        stem = path[:-3] if path.endswith(".md") else ""
        if stem and "/" not in stem and stem in self.published:
            return f'<a href="{escape(stem)}.html{"#" + escape(anchor) if anchor else ""}">{label_html}</a>'
        if not url:
            return label_html
        # Raw files and local projects are not published: keep the name only.
        return f"{label_html} (<code>{escape(url)}</code>)"

    @staticmethod
    def emphasis(text: str) -> str:
        text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|$])", r"\1", text)
        text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<![\w])__(?=\S)(.+?)(?<=\S)__(?![\w])", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])", r"<em>\1</em>", text)
        text = re.sub(r"(?<![\w])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w])", r"<em>\1</em>", text)
        text = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"<del>\1</del>", text)
        return text

    def render(self, text: str) -> str:
        held: list[str] = []

        def hold(html: str) -> str:
            held.append(html)
            return f"\x00{len(held) - 1}\x00"

        def label(raw: str) -> str:
            return self.emphasis(escape(raw, quote=False))

        text = re.sub(r"(`+)(.+?)\1", lambda m: hold(f"<code>{escape(m.group(2).strip())}</code>"), text)
        text = re.sub(r"(?<![\\$\w])\$(?! )([^$\n]+?)(?<! )\$(?![\w$])",
                      lambda m: hold(escape(m.group(0))), text)
        text = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
                      lambda m: hold(self.wiki_link(m.group(1), m.group(2))), text)
        text = re.sub(r"<(https?://[^>\s]+)>",
                      lambda m: hold(f'<a href="{escape(m.group(1))}" rel="noreferrer">{escape(m.group(1))}</a>'), text)
        text = re.sub(r"(?<!!)\[([^\]]+)\]\(((?:[^()\s]|\([^()]*\))*)\)",
                      lambda m: hold(self.md_link(label(m.group(1)), m.group(2))), text)
        text = re.sub(r"(?<![\"'=(])\bhttps?://[^\s<>\"')\]]+[^\s<>\"')\].,;:!?]",
                      lambda m: hold(f'<a href="{escape(m.group(0))}" rel="noreferrer">{escape(m.group(0))}</a>'), text)
        text = self.emphasis(escape(text, quote=False))
        while "\x00" in text:
            text = re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], text)
        return text


LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")


def split_row(line: str) -> list[str]:
    """Cells of one table row; pipes inside code or [[a|b]] stay in the cell."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, current, in_code, depth, i = [], "", False, 0, 0
    while i < len(line):
        char = line[i]
        if char == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            current += "|"
            i += 2
            continue
        if char == "`":
            in_code = not in_code
        elif line.startswith("[[", i) and not in_code:
            depth += 1
        elif line.startswith("]]", i) and not in_code and depth:
            depth -= 1
        if char == "|" and not in_code and not depth:
            cells.append(current.strip())
            current = ""
        else:
            current += char
        i += 1
    cells.append(current.strip())
    return cells


class Blocks:
    def __init__(self, inline: Inline):
        self.inline = inline
        self.ids: set[str] = set()

    def heading_id(self, text: str) -> str:
        base = slugify(text) or "section"
        slug, n = base, 1
        while slug in self.ids:
            n += 1
            slug = f"{base}-{n}"
        self.ids.add(slug)
        return slug

    def is_block_start(self, lines: list[str], i: int) -> bool:
        line = lines[i]
        return bool(
            HEADING.match(line) or FENCE.match(line) or line.lstrip().startswith(">")
            or LIST_ITEM.match(line) or RULE.match(line)
            or (line.lstrip().startswith("|") and i + 1 < len(lines) and TABLE_RULE.match(lines[i + 1]))
        )

    def render(self, text: str) -> str:
        lines = text.split("\n")
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            fence = FENCE.match(line)
            if fence:
                mark = fence.group(1)
                body = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(mark):
                    body.append(lines[i])
                    i += 1
                i += 1
                out.append(f"<pre><code>{escape(chr(10).join(body))}</code></pre>")
                continue
            heading = HEADING.match(line)
            if heading:
                level = len(heading.group(1))
                content = heading.group(2)
                hid = self.heading_id(re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
                                             lambda m: m.group(2) or m.group(1), content))
                out.append(f'<h{level} id="{hid}">{self.inline.render(content)}</h{level}>')
                i += 1
                continue
            if RULE.match(line):
                out.append("<hr>")
                i += 1
                continue
            if line.lstrip().startswith("|") and i + 1 < len(lines) and TABLE_RULE.match(lines[i + 1]):
                head = split_row(line)
                i += 2
                rows = []
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    rows.append(split_row(lines[i]))
                    i += 1
                thead = "".join(f"<th>{self.inline.render(c)}</th>" for c in head)
                tbody = "".join(
                    "<tr>" + "".join(f"<td>{self.inline.render(c)}</td>" for c in row) + "</tr>"
                    for row in rows
                )
                out.append(f'<div class="tw"><table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>')
                continue
            if line.lstrip().startswith(">"):
                quote = []
                while i < len(lines) and lines[i].lstrip().startswith(">"):
                    quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                    i += 1
                out.append(f"<blockquote>{self.render(chr(10).join(quote))}</blockquote>")
                continue
            if LIST_ITEM.match(line):
                i = self.render_list(lines, i, out)
                continue
            para = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not self.is_block_start(lines, i):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{self.inline.render(' '.join(para))}</p>")
        return "\n".join(out)

    def render_list(self, lines: list[str], i: int, out: list[str]) -> int:
        """Nested lists by indentation; continuation lines join their item."""
        items: list[tuple[int, str, str]] = []  # indent, kind, text
        while i < len(lines):
            line = lines[i]
            match = LIST_ITEM.match(line)
            if match:
                kind = "ol" if match.group(2)[0].isdigit() else "ul"
                items.append((len(match.group(1).expandtabs(4)), kind, match.group(3)))
                i += 1
                continue
            if not line.strip():
                nxt = i + 1
                if nxt < len(lines) and (LIST_ITEM.match(lines[nxt]) or
                                         (lines[nxt].startswith("  ") and lines[nxt].strip())):
                    i += 1
                    continue
                break
            if self.is_block_start(lines, i) and not line.startswith("  "):
                break
            if FENCE.match(line):
                break
            indent, kind, text = items[-1]
            items[-1] = (indent, kind, text + " " + line.strip())
            i += 1

        html: list[str] = []
        stack: list[tuple[int, str]] = []
        for indent, kind, text in items:
            while stack and indent < stack[-1][0]:
                html.append(f"</li></{stack.pop()[1]}>")
            if stack and indent == stack[-1][0] and kind != stack[-1][1]:
                html.append(f"</li></{stack.pop()[1]}>")
            if not stack or indent > stack[-1][0]:
                stack.append((indent, kind))
                html.append(f"<{kind}><li>")
            else:
                html.append("</li><li>")
            task = re.match(r"\[([ xX])\]\s+(.*)", text)
            if task:
                box = "☑" if task.group(1) in "xX" else "☐"
                text = f"{box} {task.group(2)}"
            html.append(self.inline.render(text))
        while stack:
            html.append(f"</li></{stack.pop()[1]}>")
        out.append("".join(html))
        return i


def document(title: str, body: str, bar: str, script: str = "") -> str:
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<meta name=\"robots\" content=\"noindex\">\n"
        f"<title>{escape(title)} · ML wiki</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
        f"<nav class=\"bar\">{bar}</nav>\n<main>\n{body}\n</main>\n{script}</body>\n</html>\n"
    )


PAGE_BAR = ('<a href="index.html">← Wiki index</a><span class="sp"></span>'
            '<a href="../learn/">Learn</a>')
INDEX_BAR = ('<a href="../">Course</a><span class="sp"></span><a href="../learn/">Learn</a>')


def render_page(name: str, meta: dict, body: str, published: set[str]) -> str:
    inline = Inline(published)
    blocks = Blocks(inline)
    title = meta.get("title") or name
    chips = [f'<span class="type">{escape(meta.get("type", ""))}</span>']
    dates = [f"{label} {escape(meta[key])}" for key, label in (("created", "created"), ("updated", "updated")) if meta.get(key)]
    chips.extend(f"<span>{d}</span>" for d in dates)
    chips.extend(f'<span class="tag">{escape(t)}</span>' for t in meta.get("tags", []))
    header = f'<div class="meta">{" ".join(chips)}</div>'
    html = blocks.render(body)
    if not re.match(r"\s*<h1", html):
        html = f"<h1>{escape(title)}</h1>\n{html}"
    first_h1_end = html.find("</h1>") + len("</h1>")
    html = html[:first_h1_end] + "\n" + header + html[first_h1_end:]
    sources = meta.get("sources", [])
    if sources:
        rows = []
        for source in sources:
            if re.match(r"https?://", source):
                rows.append(f'<li><a href="{escape(source)}" rel="noreferrer">{escape(source)}</a></li>')
            else:
                rows.append(f"<li><code>{escape(source)}</code></li>")
        html += (f'\n<details class="src"><summary>Sources ({len(sources)})</summary>'
                 '<p class="note">Raw files and local projects are named, not published.</p>'
                 f'<ul>{"".join(rows)}</ul></details>')
    return document(title, html, PAGE_BAR)


WIKI_LINK = re.compile(r"\[\[([^\]|#]+)[^\]]*\]\]")
COUNTED = re.compile(r"^(#{2,6}\s.*\()\d+(\)\s*)$")


def links_out(text: str, published: set[str]) -> bool:
    return any(m.group(1).strip() not in published for m in WIKI_LINK.finditer(text))


def prune_sections(body: str, published: set[str]) -> str:
    """Set a heading's "(N)" to the published pages linked below it; drop a
    section, intro text included, when it links only to unpublished pages or
    its count falls to zero."""
    lines = body.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        heading = HEADING.match(lines[i])
        if not heading or len(heading.group(1)) < 2:
            out.append(lines[i])
            i += 1
            continue
        level = len(heading.group(1))
        end = i + 1
        while end < len(lines):
            inner = HEADING.match(lines[end])
            if inner and len(inner.group(1)) <= level:
                break
            end += 1
        targets = {m.group(1).strip() for m in WIKI_LINK.finditer("\n".join(lines[i + 1:end]))}
        live = targets & published
        counted = COUNTED.match(lines[i])
        if (targets and not live) or (counted and not live):
            i = end
            continue
        out.append(f"{counted.group(1)}{len(live)}{counted.group(2)}" if counted else lines[i])
        i += 1
    return "\n".join(out)


def prune_items(line: str, published: set[str]) -> str:
    """Keep the items of a " · " list that link only to published pages, and the
    bold label in front when any item is left."""
    label = re.match(r"\s*\*\*[^*]+\*\*\s*", line)
    head = label.group(0) if label else ""
    items = [item for item in line[len(head):].split(" · ") if not links_out(item, published)]
    return head + " · ".join(items) if items else ""


def prune_index(body: str, pages: list[dict], published: set[str]) -> str:
    """Drop index entries, sentences and emptied sections that point at unpublished pages."""
    body = prune_sections(body, published)
    counts: dict[str, int] = {}
    for page in pages:
        counts[page["type"]] = counts.get(page["type"], 0) + 1

    def recount(line: str) -> str:
        def one(m: re.Match) -> str:
            kind = COUNT_LABELS.get(m.group(2))
            if kind is None:
                return m.group(0)
            n = counts.get(kind, 0)
            return f"**{n} {m.group(2)}**" if n else "\x00"
        line = re.sub(r"\*\*(\d+) ([a-z -]+?)\*\*", one, line)
        return re.sub(r"\x00(,\s*)?|,\s*\x00", "", line)

    kept: list[str] = []
    for line in body.split("\n"):
        if LIST_ITEM.match(line):
            if not links_out(line, published):
                kept.append(line)
            continue
        if HEADING.match(line) or not WIKI_LINK.search(line):
            kept.append(recount(line))
            continue
        if " · " in line:
            line = prune_items(line, published)
        sentences = re.split(r"(?<=\.)\s+(?=[A-Z*\[])", line)
        line = " ".join(s for s in sentences if not links_out(s, published))
        if line.strip():
            kept.append(line)

    out: list[str] = []
    for i, line in enumerate(kept):
        heading = HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            content = False
            for nxt in kept[i + 1:]:
                inner = HEADING.match(nxt)
                if inner and len(inner.group(1)) <= level:
                    break
                if nxt.strip() and not inner:
                    content = True
                    break
            if not content:
                continue
        out.append(line)
    return prune_sections("\n".join(out), published)


def render_index(body: str, meta: dict, pages: list[dict], published: set[str]) -> str:
    inline = Inline(published)
    html = Blocks(inline).render(prune_index(body, pages, published))
    data = json.dumps(pages, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    top = (
        f'<h1>{escape(meta.get("title", "Wiki Index"))}</h1>\n'
        f'<p class="note">{len(pages)} pages, built from the private wiki. '
        "Search matches titles, names, types and tags.</p>\n"
        '<input id="q" class="search" type="search" placeholder="Search titles and tags" '
        'autocomplete="off" aria-label="Search the wiki">\n<ul id="hits" class="hits"></ul>\n'
    )
    html = re.sub(r"^\s*<h1[^>]*>.*?</h1>", "", html, count=1, flags=re.S)
    script = (f'<script type="application/json" id="pages">{data}</script>\n'
              f"<script>{SEARCH_JS}</script>\n")
    return document(meta.get("title", "Wiki Index"), top + f'<div id="body">{html}</div>', INDEX_BAR, script)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--types", help="comma-separated page types to include, or 'all' "
                        "(default: every type but PRIVATE_TYPES, less PRIVATE_PAGES)")
    parser.add_argument("--pages", help="comma-separated page names to include as well")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    types = {t.strip() for t in args.types.split(",")} if args.types else None
    names = {p.strip().removesuffix(".md") for p in args.pages.split(",")} if args.pages else set()

    loaded = {}
    for path in sorted(args.source.glob("*.md")):
        if path.name in SKIP:
            continue
        meta, body = split_front_matter(scrub(path.read_text(encoding="utf-8")))
        loaded[path.stem] = (meta, body)

    def wanted(name: str, meta: dict) -> bool:
        if types is None and not names:
            return meta.get("type") not in PRIVATE_TYPES and name not in PRIVATE_PAGES
        return (types is not None and ("all" in types or meta.get("type") in types)) or name in names

    chosen = {n for n, (m, _) in loaded.items() if n != "index" and wanted(n, m)}
    published = chosen | ({"index"} if "index" in loaded else set())

    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)
    catalog = []
    for name in sorted(chosen):
        meta, body = loaded[name]
        html = render_page(name, meta, body, published)
        (args.out / f"{name}.html").write_text(html, encoding="utf-8")
        catalog.append({"name": name, "title": meta.get("title", name),
                        "type": meta.get("type", ""), "tags": meta.get("tags", [])})
    if "index" in loaded:
        meta, body = loaded["index"]
        (args.out / "index.html").write_text(render_index(body, meta, catalog, published), encoding="utf-8")

    leaks = [p.name for p in args.out.glob("*.html") if FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    if leaks:
        print(f"refusing: local paths left in {', '.join(leaks[:10])}", file=sys.stderr)
        shutil.rmtree(args.out)
        return 1
    size = sum(p.stat().st_size for p in args.out.iterdir())
    out = args.out.resolve()
    shown = out.relative_to(PROJECT) if out.is_relative_to(PROJECT) else out
    print(f"Built {shown}/ with {len(chosen)} pages and an index, "
          f"{size / 1024:.0f} KB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
