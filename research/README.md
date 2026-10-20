# Research companion

Interactive pages from the JEPA explorations project (October 2026), folded into the course so they travel with it and work offline.

- **Hidden Patch** (`index.html`): an 18-chapter learning site in five parts. Every chapter opens with a why-question, gives a short answer, then works through it with live labs, three takeaways, a quiz and sources. It covers predictive learning and JEPA, collapse, implicit bias, generalization, interpretability, reasoning, architectures, world models, edge efficiency and artificial life.
- **The Why Atlas** (`why-atlas.html`): about 85 people and papers linked to 14 "why" questions about how AI systems learn.
- **JEPA field map** (`jepa-field-map.html`): the JEPA paper landscape, filterable by theme, with theory first and an experiment agenda last.
- **Edge research agenda** (`agenda.html`, added 2026-10-20): eleven open questions one person can work on with a laptop or a phone, ranked, each with a first experiment written down before any code, the papers behind it, and where the course's own modules stand.

Everything is plain HTML, CSS and JavaScript with no build step and no runtime network calls. The fonts are self-hosted under `assets/fonts/` (SIL Open Font License 1.1, see `assets/fonts/LICENSE.md`). Outbound links to papers are ordinary links and only matter when you are online.

`scripts/build_reader.py` copies this folder into the offline reader as `reader/research/`, so the service worker caches it with the rest of the course, and `scripts/check_offline.py` checks it for remote dependencies and dead links.
