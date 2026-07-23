"""Shared helpers for module 08 — vision.

Device selection (MPS-first), figure paths, a consistent matplotlib style, and
small image loaders that reuse the default Hugging Face cache (nothing lands in
the repo).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PY_DIR = Path(__file__).resolve().parent.parent          # .../python
MODULE_DIR = PY_DIR.parent                               # .../08-vision
FIGURES = MODULE_DIR / "figures"
FIGURES.mkdir(exist_ok=True)

# Canonical dataset id (the bare "beans" alias trips a hub-uri bug in
# datasets 5.x / huggingface_hub, so we always use the namespaced repo).
BEANS = "AI-Lab-Makerere/beans"
BEANS_CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """MPS on Apple silicon, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Matplotlib style — quiet, readable, consistent across figures
# ---------------------------------------------------------------------------
def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "image.cmap": "gray",
        }
    )


def savefig(fig, name: str, dpi: int | None = None) -> Path:
    """Save under figures/ with tight bbox, then optimise the PNG.

    Photo-heavy figures can blow past the 300 KB budget; pass a lower ``dpi`` for
    those. We also run a lossless PIL optimise pass on every PNG.
    """
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    try:
        img = Image.open(path)
        img.save(path, optimize=True)
    except Exception:
        pass
    return path


# ---------------------------------------------------------------------------
# Sample imagery (reuses the matplotlib-bundled photo + HF cache)
# ---------------------------------------------------------------------------
def grace_hopper(size: int | None = None) -> Image.Image:
    """The classic Grace Hopper portrait bundled with matplotlib.

    ImageNet models label it "military uniform" — a good, reproducible photo
    that needs no network.
    """
    p = Path(mpl.get_data_path()) / "sample_data" / "grace_hopper.jpg"
    img = Image.open(p).convert("RGB")
    if size is not None:
        img = img.resize((size, size))
    return img


def to_gray_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 grayscale array in [0, 1]."""
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def cats_image() -> Image.Image:
    """A single cat photo (the image used throughout the transformers docs)."""
    from datasets import load_dataset

    ds = load_dataset("huggingface/cats-image")
    return ds["test"][0]["image"].convert("RGB")


def beans_samples(n: int = 3, split: str = "train", seed: int = 0):
    """Return a list of (PIL image, label-name) from the beans dataset."""
    from datasets import load_dataset

    ds = load_dataset(BEANS, split=split)
    names = ds.features["labels"].names
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    return [(ds[int(i)]["image"].convert("RGB"), names[ds[int(i)]["labels"]]) for i in idx]


def vit_processor(model_name: str):
    """Return a ViT image processor using the pure-PIL backend.

    Newer transformers default to a torchvision-backed "fast" processor; we keep
    the dependency list lean by using the PIL implementation directly.
    """
    try:
        from transformers import ViTImageProcessorPil as _P
    except ImportError:  # older transformers
        from transformers import ViTImageProcessor as _P
    return _P.from_pretrained(model_name)


def hf_offline_hint() -> None:
    """Print a note if HF_HUB_OFFLINE is set (helps debugging cold caches)."""
    if os.environ.get("HF_HUB_OFFLINE"):
        print("[note] HF_HUB_OFFLINE is set — using cache only.")
