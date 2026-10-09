# Module 06 — PyTorch/MPS lane

The primary fine-tuning lane. See the [module README](../README.md) for the full
walkthrough. Quick reference:

```bash
uv sync                              # install deps
uv run python figures_concept.py     # concept figures (~30s, no training)
uv run python sft_train.py --steps 200   # the SFT lab (~12 min on M5)
uv run trackio show                  # open the local trackio dashboard
uv run python generate_compare.py    # before/after evidence (~2 min)
uv run python solutions/ex_a_rank.py # exercise (a), verified (~8 min)
```

Layout:

- `src/common.py` — shared model/dataset/LoRA config + device pick.
- `figures_concept.py` — parameter-count bars + LoRA schematic.
- `sft_train.py` — reusable `run_sft(...)`; the exercises import it.
- `generate_compare.py` — fixed-prompt base-vs-adapter generations.
- `exercises/` — `# TODO(you):` stubs; `solutions/` — verified answers.
- `outputs/` — adapters, loss history, logs (git-ignored).

Drop to the 135M model with `export MLCOURSE_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct`.
