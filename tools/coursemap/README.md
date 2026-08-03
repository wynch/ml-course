# course map

A standalone one-page dashboard of the whole course — 11 module cards with their real measured
numbers, a figure each, links to every explorable and quiz, and a localStorage progress bar.

Build it with `uv run build_map.py` from this directory: the script inlines each module figure from
`modules/*/figures/` as a data: URI into `template.html` and writes `course-map.html` (~530 KB).

The generated `course-map.html` is gitignored — it is republished by hand as a claude.ai artifact,
not as part of the site under `wynch.github.io/ml-course`.
