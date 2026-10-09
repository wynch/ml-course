"""Shared helpers for module 06 — model/dataset config, device pick, plot style.

Kept in one place so every script (SFT lab, before/after, exercises) agrees on
which model, which dataset slice, and which LoRA settings it is talking about.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window

import matplotlib.pyplot as plt
import torch

plt.rcParams.update(
    {"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10, "axes.grid": False}
)

# --- What we fine-tune ---------------------------------------------------------
# 360M is the default (spec preference); it trains in a few minutes on the M5's
# MPS backend. Drop to 135M by exporting MLCOURSE_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
# if you want it faster still — everything downstream keys off this one constant.
import os

MODEL_ID = os.environ.get("MLCOURSE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")

# --- What we fine-tune it on ---------------------------------------------------
DATASET_ID = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "everyday-conversations"
TRAIN_SLICE = 512  # a few hundred examples — deliberately small, this is a lab

# --- LoRA defaults -------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# All linear projections in both attention and MLP — the modern default.
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def pick_device() -> str:
    """MPS on Apple silicon, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_params(model) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
