# Module 06 — PyTorch/MPS lane

The primary fine-tuning lane. See the [module README](../README.md) for the full
walkthrough. Quick reference:

```bash
uv sync                              # install deps
uv run python figures_concept.py     # concept figures (~30s, no training)
uv run python sft_train.py --steps 200   # the SFT lab (~12 min on M5)
uv run trackio show                  # open the local trackio dashboard
uv run python generate_compare.py    # before/after evidence (~2 min)
uv run python solutions/ex_a_rank.py # exercise (a), verified (~8 min); --plot-only redraws
uv run python eval_heldout.py        # held-out loss, base vs each adapter (~2 min)
uv run python svd_explorable.py      # SVD data for the explorable; --splice writes it in
```

Layout:

- `src/common.py` — shared model/dataset/LoRA config + device pick.
- `figures_concept.py` — parameter-count bars + LoRA schematic.
- `sft_train.py` — reusable `run_sft(...)`; the exercises import it.
- `generate_compare.py` — fixed-prompt base-vs-adapter generations.
- `eval_heldout.py` — loss on the 119-conversation test split, all tokens and
  replies only.
- `svd_explorable.py` — SVD of the frozen `q_proj` and its trained ΔW for the
  explorable.
- `exercises/` — `# TODO(you):` stubs; `solutions/` — verified answers.
- `outputs/` — adapters, loss history, logs (git-ignored).
- `results/` — committed numbers: held-out losses, SVD summary, a copy of each
  run's loss history, the lab's throughput summary.

Drop to the 135M model with `export MLCOURSE_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct`.
