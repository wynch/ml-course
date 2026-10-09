"""Shared paths, device selection and training hyper-parameters.

Every script imports from here so the model that gets trained, plotted and
exported is unambiguously the same one.
"""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]           # modules/04-attention-transformer
FIGURES = ROOT / "figures"
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"                              # git-ignored: checkpoints live here
CORPUS = ROOT / "corpus" / "tiny_shakespeare.txt"

for d in (FIGURES, ARTIFACTS, MODELS):
    d.mkdir(exist_ok=True)

# --- model / training hyper-parameters ------------------------------------
BLOCK_SIZE = 128
N_LAYER = 3
N_HEAD = 4
D_MODEL = 128
DROPOUT = 0.1

BATCH_SIZE = 64
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 100
LEARNING_RATE = 3e-4
WARMUP_ITERS = 150
WEIGHT_DECAY = 0.1
SEED = 1337

# checkpoints captured for the "generation over training" figure / gif
CHECKPOINT_FRACTIONS = (0.0, 0.25, 0.50, 1.0)

# prompt used everywhere we sample from the model
SAMPLE_PROMPT = "ROMEO:"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
