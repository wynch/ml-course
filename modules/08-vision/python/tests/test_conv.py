"""Tests for the from-scratch numpy convolution."""
import numpy as np
import pytest

import conv


def test_identity_is_noop():
    img = np.random.default_rng(0).random((12, 10))
    out = conv.conv2d(img, conv.KERNELS["identity"], padding="same")
    assert out.shape == img.shape
    assert np.allclose(out, img)


def test_same_padding_preserves_shape():
    img = np.random.default_rng(1).random((17, 23))
    for name, k in conv.KERNELS.items():
        out = conv.conv2d(img, k, padding="same")
        assert out.shape == img.shape, name


def test_valid_padding_shrinks():
    img = np.zeros((10, 10))
    out = conv.conv2d(img, np.ones((3, 3)), padding="valid")
    assert out.shape == (8, 8)


def test_matches_manual_sum():
    # a single interior pixel of a box filter equals the mean of its 3x3 window
    img = np.arange(25, dtype=float).reshape(5, 5)
    k = np.ones((3, 3)) / 9.0
    out = conv.conv2d(img, k, padding="valid")
    assert np.isclose(out[0, 0], img[:3, :3].mean())


def test_even_kernel_rejected():
    with pytest.raises(ValueError):
        conv.conv2d(np.zeros((5, 5)), np.ones((2, 2)))


def test_matches_scipy():
    scipy_signal = pytest.importorskip("scipy.signal")
    img = np.random.default_rng(2).random((20, 20))
    k = conv.KERNELS["sobel_x (vertical edges)"]
    ours = conv.conv2d(img, k, padding="valid")
    # scipy.correlate2d uses cross-correlation (no kernel flip), matching us
    ref = scipy_signal.correlate2d(img, k, mode="valid")
    assert np.allclose(ours, ref)
