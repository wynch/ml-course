# Module 35 code

Run everything from this folder with `uv run`. The lesson is the module's
[README](../README.md).

| File | What it does |
|---|---|
| `src/task.py` | instances, the online packer, the L1 bound, the score, first fit and best fit |
| `src/sandbox.py` | screens, runs and checks one candidate in a fresh `python -I` process |
| `src/database.py` | islands, clusters by behaviour, parents drawn by score, island resets |
| `src/prompt.py` | the prompt, and how a reply becomes a candidate program |
| `src/common.py` | paths, the laptop guard, the MLX mutator |
| `src/evolve.py` | the loop and its arms: `evolve`, `evolve_flat`, `noselect`, `noverifier` |
| `src/random_search.py` | random expression programs, same checker and budget |
| `src/library.py`, `src/oe_evaluator.py` | the OpenEvolve lane through a local `mlx_lm.server` |
| `src/analyse.py` | `results/analysis.json` and `results/scored.json` from the logs |
| `src/figures.py` | the figures in `../figures/` |
| `src/export_explorable.py` | the data block of the explorable |
| `tests/` | `uv run pytest`: results and solutions, no model |

OpenEvolve writes its checkpoints to `outputs/openevolve/`, which git ignores;
its score is in `../results/library.json`.
