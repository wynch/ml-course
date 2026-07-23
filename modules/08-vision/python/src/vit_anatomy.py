"""ViT anatomy: an image is 'tokenized' into patches.

A Vision Transformer does something almost cheeky: it chops an image into a grid
of fixed-size patches (16x16 pixels), flattens each patch, linearly projects it
to a vector, and then runs *exactly the same* transformer encoder you built for
text in modules 04/05. Patches are the "tokens"; there is no BPE, just a linear
patch embedding. This module makes that literal.
"""
from __future__ import annotations

import numpy as np


def patchify(image: np.ndarray, patch: int = 16) -> np.ndarray:
    """Cut an (H, W, C) image into a grid of non-overlapping patches.

    Returns an array of shape (n_h, n_w, patch, patch, C) — the same grid a ViT
    embeds. ``H`` and ``W`` must be divisible by ``patch``.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, None]
    H, W, C = image.shape
    if H % patch or W % patch:
        raise ValueError(f"image {H}x{W} not divisible by patch {patch}")
    n_h, n_w = H // patch, W // patch
    grid = (
        image.reshape(n_h, patch, n_w, patch, C)
        .transpose(0, 2, 1, 3, 4)  # (n_h, n_w, patch, patch, C)
    )
    return grid


def patch_sequence(image: np.ndarray, patch: int = 16) -> np.ndarray:
    """Flatten the patch grid into a sequence: (n_patches, patch*patch*C).

    This is exactly the tensor a ViT feeds to its linear patch-embedding — the
    image as a sequence of "tokens".
    """
    grid = patchify(image, patch)
    n_h, n_w = grid.shape[:2]
    return grid.reshape(n_h * n_w, -1)


def architecture_table(vit_config, text_layers: int = 30) -> list[dict]:
    """Build a side-by-side comparison of a ViT vs a text transformer.

    Returns a list of row dicts; rendered as a table in the README. ``text_layers``
    is a reference figure for a small text model from module 05.
    """
    hidden = vit_config.hidden_size
    layers = vit_config.num_hidden_layers
    heads = vit_config.num_attention_heads
    patch = vit_config.patch_size
    img = vit_config.image_size
    n_patches = (img // patch) ** 2
    return [
        {"aspect": "input unit", "text": "BPE subword token", "vit": f"{patch}x{patch} pixel patch"},
        {"aspect": "sequence length", "text": "up to context window", "vit": f"{n_patches} patches + 1 [CLS]"},
        {"aspect": "embedding", "text": "learned token lookup table", "vit": "linear projection of flattened patch"},
        {"aspect": "position info", "text": "positional encoding", "vit": "learned position embedding"},
        {"aspect": "encoder block", "text": "MHSA + MLP + LayerNorm", "vit": "MHSA + MLP + LayerNorm (identical)"},
        {"aspect": "layers", "text": f"~{text_layers}", "vit": str(layers)},
        {"aspect": "hidden size", "text": "512–4096", "vit": str(hidden)},
        {"aspect": "attention heads", "text": "8–32", "vit": str(heads)},
        {"aspect": "readout", "text": "next-token head", "vit": "[CLS] -> class head"},
    ]


if __name__ == "__main__":
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    seq = patch_sequence(img, 16)
    print("224x224 image -> patch sequence:", seq.shape, "(expected (196, 768))")
    assert seq.shape == (196, 768)
