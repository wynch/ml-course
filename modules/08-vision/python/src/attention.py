"""Attention on images: where does a ViT look?

Two views of the same thing, both computed from the model's attention weights:

* **CLS attention** — the attention the final-layer [CLS] token pays to every
  patch. Quick, single-layer.
* **Attention rollout** — multiply the (head-averaged, residual-augmented)
  attention matrices across *all* layers, following Abnar & Zuidema (2020). This
  traces how information flows from the input patches up to the [CLS] token and
  usually gives a cleaner map.

We overlay the resulting patch heatmap on the original photo.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from utils import get_device


def load_vit(model_name: str = "google/vit-base-patch16-224"):
    from transformers import ViTForImageClassification

    from utils import vit_processor

    processor = vit_processor(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name, attn_implementation="eager"  # needed to return attentions
    )
    model.eval().to(get_device())
    return processor, model


@torch.no_grad()
def get_attentions(processor, model, image: Image.Image):
    """Run the image through the ViT; return (attentions, logits, grid_size).

    ``attentions`` is a tuple of (n_layers) tensors, each (1, heads, T, T) where
    T = n_patches + 1 (the +1 is the [CLS] token at index 0).
    """
    device = get_device()
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model(**inputs, output_attentions=True)
    attns = [a.detach().cpu() for a in out.attentions]
    n_patches = attns[0].shape[-1] - 1
    grid = int(round(n_patches**0.5))
    return attns, out.logits.detach().cpu(), grid


def cls_attention_map(attns, grid: int) -> np.ndarray:
    """Head-averaged [CLS]->patches attention from the last layer -> (grid, grid)."""
    last = attns[-1][0]                 # (heads, T, T)
    cls_to_patches = last[:, 0, 1:]     # (heads, n_patches)
    m = cls_to_patches.mean(0).numpy()  # average over heads
    return m.reshape(grid, grid)


def attention_rollout(attns, grid: int) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020) -> (grid, grid) heatmap.

    At each layer: average heads, add the identity (residual connection), and
    row-normalise; then multiply the layer matrices together. Row 0 of the
    product is how the [CLS] token attends to every input patch.
    """
    result = torch.eye(attns[0].shape[-1])
    for a in attns:
        a = a[0].mean(0)                       # average heads -> (T, T)
        a = a + torch.eye(a.shape[-1])         # residual connection
        a = a / a.sum(dim=-1, keepdim=True)    # renormalise rows
        result = a @ result
    mask = result[0, 1:].numpy()               # [CLS] row, drop CLS->CLS
    return mask.reshape(grid, grid)


def overlay_heatmap(image: Image.Image, heat: np.ndarray, size: int = 224) -> np.ndarray:
    """Upsample ``heat`` to ``size`` and return an RGBA-friendly [0,1] map."""
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(
        (size, size), resample=Image.BICUBIC
    )
    return np.asarray(heat_img, dtype=np.float32) / 255.0


def top_label(model, logits) -> str:
    idx = int(logits.argmax(-1))
    return model.config.id2label.get(idx, str(idx))


if __name__ == "__main__":
    from utils import grace_hopper

    processor, model = load_vit()
    attns, logits, grid = get_attentions(processor, model, grace_hopper())
    print("grid", grid, "layers", len(attns), "pred:", top_label(model, logits))
    r = attention_rollout(attns, grid)
    print("rollout map", r.shape, "range", round(float(r.min()), 4), round(float(r.max()), 4))
