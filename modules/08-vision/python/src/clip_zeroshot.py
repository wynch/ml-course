"""Zero-shot classification with CLIP — no fine-tuning at all.

CLIP was pretrained to match images with their captions. That single objective,
at internet scale, teaches it enough that you can classify *new* categories just
by writing them as text prompts ("a photo of a healthy bean leaf") and asking
which prompt each image is closest to. We run it on the same beans test set and
compare against our fine-tuned ViT-tiny — a concrete lesson in pretraining scale
vs task-specific fine-tuning.
"""
from __future__ import annotations

import numpy as np
import torch

from utils import BEANS, get_device

CLIP_NAME = "openai/clip-vit-base-patch32"

# Human-readable prompts, one per beans class (order matches the label ids).
# We describe the *visible symptoms* rather than the agricultural term — CLIP
# has no idea what "angular leaf spot" means, but it has seen "grey dead
# patches" and "orange rust spots". Even so, it barely clears the 33% chance
# baseline: a fair, honest attempt that still loses badly to fine-tuning.
PROMPTS = [
    "a leaf with grey angular dead patches",
    "a leaf with small orange-brown rust pustules",
    "a uniformly green healthy leaf",
]
SHORT = ["angular_leaf_spot", "bean_rust", "healthy"]


def _features(out):
    """Extract the projected embedding tensor across transformers versions.

    Older versions returned a tensor directly; 5.x returns an output object
    whose ``pooler_output`` holds the projected image/text embedding.
    """
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_NAME).eval().to(get_device())
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    return model, processor


@torch.no_grad()
def zero_shot_beans(split: str = "test", n: int | None = None, seed: int = 0):
    """Classify beans images zero-shot; return dict with accuracy + sim matrix."""
    from datasets import load_dataset

    device = get_device()
    model, processor = load_clip()
    ds = load_dataset(BEANS, split=split)
    class_names = ds.features["labels"].names

    if n is not None:
        rng = np.random.default_rng(seed)
        idx = sorted(int(i) for i in rng.choice(len(ds), size=n, replace=False))
        ds = ds.select(idx)

    images = [ex.convert("RGB") for ex in ds["image"]]
    y_true = np.asarray(ds["labels"])

    # Encode all prompts once.
    text_inp = processor(text=PROMPTS, return_tensors="pt", padding=True).to(device)
    text_emb = _features(model.get_text_features(**text_inp))
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # Encode images in batches.
    sims = []
    B = 32
    for i in range(0, len(images), B):
        batch = images[i : i + B]
        img_inp = processor(images=batch, return_tensors="pt").to(device)
        img_emb = _features(model.get_image_features(**img_inp))
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims.append((img_emb @ text_emb.T).cpu())
    sim = torch.cat(sims).numpy()          # (n_images, n_prompts) cosine sims
    y_pred = sim.argmax(1)
    acc = float((y_pred == y_true).mean())
    return {
        "accuracy": acc,
        "sim": sim,
        "y_true": y_true,
        "y_pred": y_pred,
        "class_names": class_names,
        "prompts": PROMPTS,
    }


if __name__ == "__main__":
    out = zero_shot_beans()
    print(f"CLIP zero-shot beans accuracy: {out['accuracy']:.4f} over {len(out['y_true'])} test images")
