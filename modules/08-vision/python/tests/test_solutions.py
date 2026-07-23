"""Verify the reference solutions.

The (c) solution actually spins up the HF Trainer on a downloaded dataset, so it
is marked slow and trains only a couple of steps here. Run everything with:
    uv run pytest
or skip the slow one with:
    uv run pytest -m "not slow"
"""
import numpy as np
import pytest

import solution_a_patchify as sol_a


def test_solution_a_matches_reference():
    img = np.arange(48 * 32 * 3).reshape(48, 32, 3)
    seq = sol_a.patchify(img, 16)
    assert seq.shape == (6, 768)
    assert np.array_equal(seq[0], img[:16, :16, :].reshape(-1))
    # row-major patch order: second patch is the next block to the right
    assert np.array_equal(seq[1], img[:16, 16:32, :].reshape(-1))


def test_solution_a_agrees_with_src():
    import vit_anatomy

    img = np.random.default_rng(0).integers(0, 255, (32, 48, 3))
    a = sol_a.patchify(img, 16)
    b = vit_anatomy.patch_sequence(img, 16)
    assert np.array_equal(a, b)


@pytest.mark.slow
def test_solution_c_trains_a_few_steps():
    """Smoke-test the dataset swap: it must start and take a few train steps."""
    sol_c = pytest.importorskip("solution_c_swap_dataset")
    acc = sol_c.main(epochs=1, max_steps=3)
    assert 0.0 <= acc <= 1.0
