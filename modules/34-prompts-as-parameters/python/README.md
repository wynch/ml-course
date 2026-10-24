# Module 34, the Python lane

Run everything from this folder; see the module's [README](../README.md).

| File | What |
|---|---|
| `src/common.py` | paths, the laptop guard, results I/O, the optimiser model |
| `src/task.py` | TREC, the three splits, the fixed template and scorer |
| `src/baseline.py` | the baseline and the hand-written instruction |
| `src/optimise.py` | APE, OPRO, TextGrad-style and GEPA-style search, one budget, resumable |
| `src/library.py` | the same task in DSPy, tuned by `dspy.GEPA` |
| `src/analyse.py`, `src/figures.py`, `src/export_explorable.py` | numbers, figures and the explorable's data, from the committed results |
| `run_all.sh` | all twelve runs, chunk by chunk |
| `tests/` | `uv run pytest`: no model |
