"""Tests for patchify and the anatomy helpers."""
import numpy as np
import pytest

import vit_anatomy


def test_patch_sequence_shape():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    seq = vit_anatomy.patch_sequence(img, 16)
    assert seq.shape == (196, 768)


def test_patchify_grid_shape():
    img = np.zeros((64, 32, 3))
    grid = vit_anatomy.patchify(img, 16)
    assert grid.shape == (4, 2, 16, 16, 3)


def test_patchify_top_left_block():
    img = np.arange(32 * 32 * 3).reshape(32, 32, 3)
    grid = vit_anatomy.patchify(img, 16)
    assert np.array_equal(grid[0, 0], img[:16, :16, :])


def test_patchify_rejects_indivisible():
    with pytest.raises(ValueError):
        vit_anatomy.patchify(np.zeros((30, 30, 3)), 16)


def test_grayscale_promoted_to_channel():
    grid = vit_anatomy.patchify(np.zeros((32, 32)), 16)
    assert grid.shape == (2, 2, 16, 16, 1)
