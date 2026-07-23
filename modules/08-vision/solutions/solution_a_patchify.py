"""Reference solution for exercise (a) — patchify."""
from __future__ import annotations

import numpy as np


def patchify(image: np.ndarray, patch: int = 16) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, None]
    H, W, C = image.shape
    if H % patch or W % patch:
        raise ValueError(f"image {H}x{W} not divisible by patch {patch}")
    n_h, n_w = H // patch, W // patch
    return (
        image.reshape(n_h, patch, n_w, patch, C)
        .transpose(0, 2, 1, 3, 4)          # (n_h, n_w, patch, patch, C)
        .reshape(n_h * n_w, patch * patch * C)
    )


if __name__ == "__main__":
    img = np.arange(32 * 32 * 3).reshape(32, 32, 3)
    seq = patchify(img, 16)
    assert seq.shape == (4, 768)
    assert np.array_equal(seq[0], img[:16, :16, :].reshape(-1))
    print("patchify OK:", seq.shape)
