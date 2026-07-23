"""Cross-language verification: the Zig trainer must emit the *same* merges.

This is the teaching centerpiece of the module. Two independent BPE
implementations — one in Python, one in Zig — train on the same corpus with
the same deterministic tie-break, so their merge lists must match byte for byte.
"""

import os
import shutil
import subprocess

import pytest
from bpe import BPETokenizer

HERE = os.path.dirname(__file__)
CORPUS = os.path.abspath(os.path.join(HERE, "..", "..", "corpus", "input.txt"))
ZIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "zig"))
ZIG_BIN = os.path.join(ZIG_DIR, "zig-out", "bin", "bpe")

# Keep the test brisk: 64 merges on the full corpus is a few seconds in Python
# and a blink in Zig, and is more than enough to catch any divergence.
N_MERGES = 64

pytestmark = pytest.mark.skipif(
    shutil.which("zig") is None, reason="zig compiler not on PATH"
)


@pytest.fixture(scope="module")
def zig_binary():
    """Build the Zig trainer in ReleaseFast and return the binary path."""
    subprocess.run(
        ["zig", "build", "-Doptimize=ReleaseFast"],
        cwd=ZIG_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    assert os.path.exists(ZIG_BIN), "zig build did not produce zig-out/bin/bpe"
    return ZIG_BIN


def test_zig_merges_match_python(zig_binary, tmp_path):
    # Python side
    data = open(CORPUS, "rb").read()
    py = BPETokenizer.train(data, N_MERGES)

    # Zig side
    out = tmp_path / "zig_merges.txt"
    proc = subprocess.run(
        [zig_binary, CORPUS, str(N_MERGES), str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "zig: trained" in proc.stderr
    zig_merges = BPETokenizer.load_merges(str(out))

    assert len(zig_merges) == N_MERGES
    assert zig_merges == py.merges, (
        "Zig and Python merge lists diverged!\n"
        f"first mismatch around: py={py.merges[:5]} zig={zig_merges[:5]}"
    )
