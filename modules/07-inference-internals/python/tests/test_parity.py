"""The green gate for module 07: the Zig engine must reproduce PyTorch's logits.

This builds the Zig binary (if needed), runs both forwards on a fixed prompt and
asserts the max absolute logit difference is below 1e-3 — f32 arithmetic noise.
It also runs the engine's selfcheck: cached decode against a cache-free full recompute.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.parity import ARTIFACTS, TOL, ZIG_BIN, ZIG_DIR, ensure_zig_built, run_parity


def test_logits_parity_with_pytorch():
    # no figure needed in the test path; keep it fast and headless
    maxdiff = run_parity(make_fig=False)
    assert maxdiff < TOL, f"logit parity failed: max|Δ| = {maxdiff:.3e} >= {TOL:g}"


def test_kv_cache_self_consistency():
    """Cached decode must match a cache-free full recompute at every position."""
    ensure_zig_built()
    res = subprocess.run(
        [str(ZIG_BIN), "selfcheck", "--artifacts", str(ARTIFACTS)],
        cwd=ZIG_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "selfcheck" in res.stdout
    # exit code 0 already means the internal assert (max|Δ| < 1e-4) passed
