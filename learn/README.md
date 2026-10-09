# Learn with Claude

A page for learning from a phone: the course, a compass snapshot, the wiki, and a
pack that lets a claude.ai chat with no tools run compass sessions. Published with
the rest of `course/` at https://wynch.github.io/ml-course/learn/.

| Path | What | Made by |
|---|---|---|
| `learn/index.html` | the page: Next box, links, pack copy button, ready prompts, doors | hand-written; regions between `gen` markers by `build_pack.py` |
| `learn/pack.md` | the pack Claude reads: compass rules, spine, territory, a status table, Next box, quizzes, URLs; no log rows, since the log stays on the Mac; 39 KB, about 10,000 tokens | `build_pack.py` |
| `learn/project-instructions.md` | text to paste as the claude.ai Project instructions | hand-written |
| `compass/index.html` | copy of the private `compass/index.html`, identical after three added header lines and two scrubs (home paths, a private note's name) | `build_pack.py` |
| `wiki/` | static wiki reader, 320 self-contained pages plus a searchable index; experiments, ideas and the analyses of the local projects left out | `build_wiki.py` |

The pack, the page's Next box and the compass copy are snapshots, dated the day of
the build. Rebuild them before a session away from the Mac.

Publish with `course/deploy.sh <message> [--push]`, from the private project root.

## Rebuild

From the private project root, wiki first (the pack links to built wiki pages):

```sh
python3 course/scripts/build_wiki.py
python3 course/scripts/build_pack.py
```

`build_wiki.py` leaves out by default the types `experiment` and `idea` and the
analysis pages about the local projects; `--types concept,guide,...` and
`--pages name,...` choose the pages instead, and `--types all` builds every page.
Links to pages left out show as plain text. It skips `log.md`, names raw files without
linking them, rewrites home paths to project-relative ones (`../ml-lab/...`), and
refuses to write if a home path or vault link is left. `build_pack.py` needs `uv`
for the compass tools.

## A session from the phone

1. Open the page; tap Copy pack (or Save file) and add it to the claude.ai
   Project "ML compass", whose instructions are `project-instructions.md`.
2. Copy a ready prompt (`next`, `walk X`, `test`) into a chat in that Project.
3. At the end Claude prints log rows (and any spine YAML); paste them into the
   Apple Notes note "ML compass log".
4. At the Mac, an agent applies the rows to `compass/log.jsonl` in a normal sitting.
