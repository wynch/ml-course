# course map

A standalone one-page dashboard of the whole course — 11 module cards with their real measured
numbers, a figure each, links to every explorable and quiz, and a localStorage progress bar.

Build it with `uv run build_map.py` from this directory (or `uv run tools/coursemap/build_map.py`
from the repo root): the script inlines each module figure from `modules/*/figures/` as a data: URI
into `template.html` and writes the finished page (~530 KB) twice, identically:

- **`map.html` at the repo root** — the published page, tracked in git and served by GitHub Pages
  at <https://wynch.github.io/ml-course/map.html>. This is the maintained home of the map.
- **`course-map.html` here** — a scratch copy for one-off sharing (it used to be republished by hand
  as a claude.ai artifact). Gitignored.

`template.html` is the source of truth: edit it, then rebuild. Never hand-edit a generated page.
