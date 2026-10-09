"""Exercise (a) — implement patchify.

A Vision Transformer's very first step is to cut an image into a grid of
non-overlapping square patches and flatten each one into a vector. Implement
that here from scratch with numpy. Get the shapes right and the rest of a ViT is
"just" the transformer you already built in modules 04/05.

Run a self-check with:
    uv run python ../exercises/exercise_a_patchify.py
(or point pytest at tests/test_solutions.py once you're done).
"""
from __future__ import annotations

import numpy as np


def patchify(image: np.ndarray, patch: int = 16) -> np.ndarray:
    """Cut ``image`` into non-overlapping patches.

    Parameters
    ----------
    image : array of shape (H, W, C)  — H and W must be divisible by ``patch``.
    patch : side length of each square patch.

    Returns
    -------
    array of shape (n_patches, patch * patch * C), where
    n_patches = (H // patch) * (W // patch). Patch order is row-major
    (left-to-right, top-to-bottom), and within a patch the flattened order is
    (row, col, channel).

    Hints
    -----
    * H, W, C = image.shape
    * n_h, n_w = H // patch, W // patch
    * reshape to (n_h, patch, n_w, patch, C), then transpose so the two patch
      axes sit next to each other: (n_h, n_w, patch, patch, C)
    * finally reshape to (n_h * n_w, patch * patch * C)
    """
    # TODO(you): implement the four steps in the hints above.
    raise NotImplementedError


if __name__ == "__main__":
    # A 32x32x3 image with patch=16 must give 4 patches of length 16*16*3 = 768.
    img = np.arange(32 * 32 * 3).reshape(32, 32, 3)
    seq = patchify(img, 16)
    assert seq.shape == (4, 768), f"expected (4, 768), got {seq.shape}"
    # top-left patch must equal the top-left 16x16 block, flattened
    assert np.array_equal(seq[0], img[:16, :16, :].reshape(-1))
    print("patchify OK:", seq.shape)
