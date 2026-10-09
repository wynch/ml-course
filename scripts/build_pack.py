#!/usr/bin/env python3
"""Refresh the Learn page snapshots from the private project's live files.

Run from the private project root (needs `uv` for the compass tools):

    python3 course/scripts/build_pack.py

Writes three things, all dated today:
- learn/pack.md, the one file a chat Claude with no tools reads; it leaves out
  compass/log.jsonl and carries each spine entry's status instead;
- the generated regions of learn/index.html (Next box, doors);
- compass/index.html, a copy of the compass page, and compass/README.md.
"""

from __future__ import annotations

from datetime import date
from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reader import MODULES  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SITE = "https://wynch.github.io/ml-course/"
TODAY = date.today().isoformat()
TOOLS = ["uv", "run", "--project", "compass/tools"]
# Home paths in the compass page, made relative to the project root.
HOME = [
    (re.compile(r"(?:/Users/[^/\s\"'<]+|~)/Perso/ml/"), ""),
    (re.compile(r"(?:/Users/[^/\s\"'<]+|~)/Perso/"), "../"),
]
# A private Obsidian note named in the compass page. The pattern captures the
# title so this public script never holds it; outputs are checked against it.
NOTE = re.compile(r'the paper list in Obsidian named "([^"]+)"')
NOTE_TEXT = "a paper list in an Obsidian note"
LEAK = re.compile(r'/Users/|~/Perso|Obsidian named "')
SHELL = ('<!doctype html>\n<meta charset="utf-8">\n'
         '<meta name="viewport" content="width=device-width, initial-scale=1">\n')

NO_TOOLS = """\
You are reading this in a claude.ai chat with no tools. You cannot run lint,
`build.py` or git, and you cannot open files or links. Your job is the session
verbs exactly as the rules below define them: `next`, `walk X`, `angles`, `test`
and `intake`. Rule 5's closing steps (lint, `--apply`, the version bump, commit)
are Vincent's, at home; skip them.

The log (`log.jsonl`: his questions, test answers and slips) stays on the Mac and
is not in this pack. You do not need it: the Next box in Part 5 already orders
`test`, and Part 4 gives each entry's status and `since:` date. At the end of
every session, print the new `log.jsonl` rows in one code block, one JSON object
per line, in the exact format below, for Vincent to paste into his Apple Notes
note "ML compass log". Give each row the real date of the session: a test on a
later day may turn an entry `owned`, so the date is the proof. If the session adds
or changes a spine entry or a territory cell, print that as exact YAML in a second
block. Never invent a def and never change a status on your own: only Vincent's
words count, and a def enters only after he confirms it. The Next box is dated;
if he pastes in log rows newer than it, they win.

The row format, with one made-up row. It is an example only, not a real result:

```json
{"date": "2026-01-01", "type": "test", "entry": "example-notion", "result": "fail", "question": "Example question.", "answer": "Example answer, in his words.", "correct": "Example of the right answer.", "slip": "Example: what went wrong, in his words."}
```

A test row may also carry `quiz` (the id of the course quiz he took after a
fail; that row changes no status) and `same_day: true` (a second test of the same
notion on the same day; it changes no status). A question row has `date`, `type`
set to `question`, `q`, `text`, `entry` (a spine key or null) and `status` set to
`open`.
"""

# Said after the Compass rules, which assume the log is at hand.
NO_LOG = """\
_Without the log: rule 4's "first unused quiz" cannot be checked here. Ask Vincent
which quizzes of the notion's `test:` list he has taken, and give the first one
he has not._
"""


def run(command: list[str]) -> str:
    return subprocess.run(command, cwd=PROJECT, check=True, capture_output=True, text=True).stdout


def compass_section() -> str:
    text = (PROJECT / "AGENTS.md").read_text()
    start = text.index("\n## Compass\n") + 1
    end = text.index("\n## ", start + 1)
    return text[start:end].rstrip() + "\n"


def load_spine() -> dict:
    code = "import json,yaml;print(json.dumps(yaml.safe_load(open('compass/spine.yaml')),default=str))"
    return json.loads(run([*TOOLS, "python", "-c", code]))


def door_url(path: str) -> str | None:
    """Public URL of a spine door, or None when it lives outside course/."""
    if not path.startswith("course/"):
        return None
    rest = path[len("course/"):]
    module = re.match(r"modules/([^/]+)/README\.md", rest)
    if module:
        return f"{SITE}reader/#/{module.group(1)}"
    return SITE + rest.split("#")[0]


def as_list(value) -> list[str]:
    if not value:
        return []
    return value if isinstance(value, list) else [value]


QUESTION = re.compile(
    r"""["']?id["']?\s*:\s*["'](q\d+)["'][\s\S]*?["']?topic["']?\s*:\s*["']([^"']*)["']"""
)


def quizzes() -> list[dict]:
    titles = {m[0].split("-")[0]: (m[1], m[0]) for m in MODULES}
    found = []
    for path in sorted((ROOT / "quizzes").glob("*.html")):
        if path.stem == "index":
            continue
        title, slug = titles.get(path.stem, (path.stem, path.stem))
        questions = QUESTION.findall(path.read_text())
        found.append({"id": path.stem, "title": title, "slug": slug,
                      "file": path.name, "questions": questions})
    return found


def fenced(lang: str, text: str) -> str:
    return f"```{lang}\n{text.rstrip()}\n```\n"


def status_section(spine: dict) -> str:
    """Status of each spine entry and counts per status; nothing from the log."""
    counts: dict[str, int] = {}
    rows = ["| Term | Status | Since |", "|---|---|---|"]
    for key, entry in spine.items():
        status = entry.get("status", "")
        counts[status] = counts.get(status, 0) + 1
        rows.append(f"| {entry.get('term', key)} | {status} | {entry.get('since', '')} |")
    total = ", ".join(f"{status} {n}" for status, n in sorted(counts.items()))
    return f"{len(spine)} entries: {total}.\n\n" + "\n".join(rows) + "\n"


def build_pack(next_box: str, spine: dict, quiz_list: list[dict]) -> str:
    parts = [
        f"# ML compass pack · {TODAY}\n",
        "Built by `course/scripts/build_pack.py` from the live files of Vincent's "
        "private ML project. Seven parts, each with a one-line header saying what "
        "it is and where it comes from.\n",
        "## Part 1 · Compass rules\n",
        "_The whole \"Compass\" section of the project's `AGENTS.md`, unchanged, "
        "between a note for a chat with no tools and one line on the missing log._\n",
        NO_TOOLS,
        compass_section(),
        NO_LOG,
        "## Part 2 · spine.yaml\n",
        "_`compass/spine.yaml`, whole: every notion Vincent has defined, in his words._\n",
        fenced("yaml", (PROJECT / "compass/spine.yaml").read_text()),
        "## Part 3 · territory.yaml\n",
        "_`compass/territory.yaml`, whole: the seven angles and the map cells._\n",
        fenced("yaml", (PROJECT / "compass/territory.yaml").read_text()),
        "## Part 4 · Status\n",
        "_From `compass/spine.yaml`: each entry's status and `since:` date, and the "
        "count per status. Nothing from `log.jsonl`, which stays on the Mac._\n",
        status_section(spine),
        "## Part 5 · Next box\n",
        f"_Printed by `build.py --next` on {TODAY}._\n",
        fenced("text", next_box),
        "## Part 6 · Course quizzes\n",
        "_From `course/quizzes/`. A spine `test:` id `quiz-<quiz>-q<n>` means question "
        f"n of that quiz. URL: `{SITE}quizzes/<file>`; the page shows all ten "
        "questions, so name the question number and its topic._\n",
    ]
    for quiz in quiz_list:
        parts.append(f"### quiz-{quiz['id']} · {quiz['title']}\n")
        parts.append(f"{SITE}quizzes/{quiz['file']}\n")
        parts.append("\n".join(f"- `quiz-{quiz['id']}-{qid}` {topic}" for qid, topic in quiz["questions"]) + "\n")
    parts.append("## Part 7 · URL map\n")
    parts.append("_Where things are on the public site, so you can give Vincent links. "
                 "Doors outside `course/` are private and have no URL._\n")
    rows = [
        ("Learn page", f"{SITE}learn/"),
        ("Course home", SITE),
        ("Reader (all lessons)", f"{SITE}reader/"),
        ("Reader, one module", f"{SITE}reader/#/<module-slug>, e.g. {SITE}reader/#/01-autograd"),
        ("Explorables", f"{SITE}explorables/index.html"),
        ("One explorable", f"{SITE}explorables/<file>; spine door `course/explorables/X` maps to it"),
        ("Quizzes", f"{SITE}quizzes/index.html"),
        ("Course map", f"{SITE}map.html"),
        ("Compass page (snapshot)", f"{SITE}compass/"),
        ("Wiki index, with search", f"{SITE}wiki/"),
        ("One wiki page", f"{SITE}wiki/<name>.html; spine `read: wiki/<name>.md` maps to it"),
        ("This pack", f"{SITE}learn/pack.md"),
    ]
    parts.append("| What | URL |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows) + "\n")
    parts.append("\nModules: " + ", ".join(f"`{m[0]}` {m[1]}" for m in MODULES) + ".\n")
    parts.append("\nDoors per spine entry:\n")
    parts.append("| Entry | Status | Door | Wiki |\n|---|---|---|---|")
    for key, entry in spine.items():
        doors = [(door_url(d) + (f" (section `#{d.split('#')[1]}`)" if "#" in d else ""))
                 if door_url(d) else f"private: `{d}`" for d in as_list(entry.get("see"))]
        reads = [f"{SITE}wiki/{Path(r).stem}.html" for r in as_list(entry.get("read"))
                 if (ROOT / "wiki" / f"{Path(r).stem}.html").exists()]
        parts.append(f"| {entry.get('term', key)} | {entry.get('status', '')} | "
                     f"{'<br>'.join(doors) or 'none'} | {'<br>'.join(reads) or 'none'} |")
    return "\n".join(parts) + "\n"


def fill(html: str, region: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- gen:{region} -->)[\s\S]*?(<!-- /gen:{region} -->)")
    if not pattern.search(html):
        raise SystemExit(f"learn/index.html has no gen:{region} region")
    return pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", html)


def build_learn_page(next_box: str, spine: dict) -> None:
    page = ROOT / "learn/index.html"
    html = page.read_text()
    html = fill(html, "next", f'<p class="when">Printed {TODAY}</p>\n<pre>{escape(next_box)}</pre>')
    public, private = [], []
    for key, entry in spine.items():
        term = escape(entry.get("term", key))
        status = escape(entry.get("status", ""))
        urls = [u for u in (door_url(d) for d in as_list(entry.get("see"))) if u]
        if urls:
            links = " ".join(f'<a href="{escape(u.replace(SITE, "../"))}">door</a>' for u in urls)
            public.append(f'<li><span class="t">{term}</span> <span class="s {status}">{status}</span> {links}</li>')
        else:
            where = ", ".join(as_list(entry.get("see"))) or "no door yet"
            private.append(f'<li><span class="t">{term}</span> <span class="s {status}">{status}</span> '
                           f'<code>{escape(where)}</code></li>')
    html = fill(html, "doors", f'<ul class="doors">{"".join(public)}</ul>')
    html = fill(html, "private", f'<ul class="doors">{"".join(private)}</ul>')
    html = fill(html, "built", f"<span>Built {TODAY}</span>")
    instructions = (ROOT / "learn/project-instructions.md").read_text()
    html = fill(html, "instructions", f'<pre id="instructions">{escape(instructions.rstrip())}</pre>')
    page.write_text(html)


PRIVATE: set[str] = set()  # note titles found in the source page


def copy_compass() -> str:
    source = PROJECT / "compass/index.html"
    target = ROOT / "compass/index.html"
    target.parent.mkdir(exist_ok=True)
    text = source.read_text()
    for pattern, replacement in HOME:
        text = pattern.sub(replacement, text)
    PRIVATE.update(NOTE.findall(text))
    text = NOTE.sub(NOTE_TEXT, text)
    # The page is a fragment written for a host that adds the document shell.
    # Served bare it falls into quirks mode at 980 px, so the copy gets the
    # shell in front; past the shell, only the two scrubs differ.
    wrapped = not text.lstrip().lower().startswith("<!doctype")
    target.write_text(SHELL + text if wrapped else text)
    version = re.search(r"version (\d+\.\d+)", text)
    label = version.group(1) if version else "unknown"
    (ROOT / "compass/README.md").write_text(
        "# Compass page, published copy\n\n"
        f"`index.html` here is a copy of the private project's `compass/index.html`, "
        f"version {label}, taken {TODAY} by `scripts/build_pack.py`.\n\n"
        + ("It is identical after three added header lines and two scrubs. "
           "The three lines (doctype, charset, viewport) make a phone render it at its "
           "own width. " if wrapped else "It is identical after two scrubs. ") +
        "The first makes home paths relative to the project root "
        "(`compass/index.html`, `../ml-lab/crystallizer`); the second replaces the "
        "title of a private Obsidian note with \"an Obsidian note\".\n\n" +
        "The source of truth is `compass/index.html` in the private project. Never "
        "edit this copy; refresh it with the script.\n"
    )
    return label


def main() -> int:
    next_box = run([*TOOLS, "compass/tools/build.py", "--next"]).rstrip()
    spine = load_spine()
    quiz_list = quizzes()
    if not (ROOT / "wiki/index.html").exists():
        print("warning: course/wiki/ is not built; wiki links in the pack are empty", file=sys.stderr)
    pack = build_pack(next_box, spine, quiz_list)
    (ROOT / "learn").mkdir(exist_ok=True)
    (ROOT / "learn/pack.md").write_text(pack)
    build_learn_page(next_box, spine)
    version = copy_compass()
    for path in (ROOT / "learn/pack.md", ROOT / "learn/index.html", ROOT / "compass/index.html"):
        text = path.read_text()
        if LEAK.search(text) or any(name in text for name in PRIVATE):
            print(f"refusing: a home path or note name is in {path.relative_to(PROJECT)}", file=sys.stderr)
            return 1
    print(f"pack.md {len(pack.encode()) / 1024:.1f} KB (~{len(pack) // 4} tokens), "
          f"{sum(len(q['questions']) for q in quiz_list)} quiz questions, "
          f"compass v{version}, dated {TODAY}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
