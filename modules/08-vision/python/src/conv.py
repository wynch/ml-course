"""2D convolution from scratch (numpy) + classic hand-designed kernels.

This is the conceptual root of the whole module: a convolution slides a small
weight grid (a *kernel*) over an image and, at every position, computes a
weighted sum of the neighbourhood. Hand-picked kernels detect edges, blur, or
sharpen. The punchline of the CNN section is that training *discovers* kernels
like these on its own.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------
def conv2d(image: np.ndarray, kernel: np.ndarray, *, padding: str = "same") -> np.ndarray:
    """Cross-correlate a 2D grayscale ``image`` with ``kernel``.

    We implement the machine-learning convention (cross-correlation: the kernel
    is *not* flipped — this is exactly what ``torch.nn.Conv2d`` computes).

    Parameters
    ----------
    image : (H, W) float array
    kernel : (kH, kW) float array, odd side lengths
    padding : "same" keeps the output the same shape (zero-padded border);
              "valid" only covers positions where the kernel fully overlaps.

    Returns
    -------
    (H, W) array for "same", or (H-kH+1, W-kW+1) for "valid".
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("conv2d expects 2D image and 2D kernel")
    kH, kW = kernel.shape
    if kH % 2 == 0 or kW % 2 == 0:
        raise ValueError("kernel sides must be odd")

    if padding == "same":
        pad = ((kH // 2, kH // 2), (kW // 2, kW // 2))
        img = np.pad(image, pad, mode="constant")
    elif padding == "valid":
        img = image
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    H, W = img.shape
    outH, outW = H - kH + 1, W - kW + 1
    out = np.empty((outH, outW), dtype=np.float64)
    # Straightforward sliding window — clear over clever. Vectorised over the
    # kernel elements so it stays fast enough for a 600px photo.
    for i in range(outH):
        for j in range(outW):
            region = img[i : i + kH, j : j + kW]
            out[i, j] = np.sum(region * kernel)
    return out


# ---------------------------------------------------------------------------
# Classic hand-designed kernels
# ---------------------------------------------------------------------------
def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


KERNELS: dict[str, np.ndarray] = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
    "sobel_x (vertical edges)": np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float
    ),
    "sobel_y (horizontal edges)": np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float
    ),
    "laplacian (all edges)": np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float
    ),
    "gaussian blur": gaussian_kernel(5, 1.2),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
}


def normalize01(a: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1] for display (edge maps are signed)."""
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


if __name__ == "__main__":
    # tiny smoke test
    img = np.arange(25, dtype=float).reshape(5, 5)
    ident = conv2d(img, KERNELS["identity"], padding="same")
    assert np.allclose(ident, img), "identity kernel must be a no-op"
    print("conv2d identity round-trips OK; output shape", ident.shape)
