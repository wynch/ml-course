"""Parity: does the framework-free Zig engine compute the SAME logits as
PyTorch?

Both sides read the identical `tiny_gpt_weights.bin`. For a fixed prompt we:
  1. run the PyTorch reference forward and take the last position's 65 logits;
  2. run `tiny-gpt logits` (Zig) for the same prompt, which dumps its 65 logits
     as raw little-endian f32;
  3. assert max|Δ| < 1e-3 (comfortably inside f32 arithmetic noise);
  4. draw a parity scatter + an error histogram.

If this passes, every subsequent lab (KV cache, int8) is trustworthy: the
engine really is the module-04 model, just without the framework.

Run:  uv run python scripts/parity.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ref_model import encode, load_reference

MODULE = Path(__file__).resolve().parents[2]           # modules/07-inference-internals
ARTIFACTS = MODULE.parent / "04-attention-transformer" / "artifacts"
ZIG_DIR = MODULE / "zig"
ZIG_BIN = ZIG_DIR / "zig-out" / "bin" / "tiny-gpt"
FIGURES = MODULE / "figures"

PROMPT = "ROMEO: But soft, what light through yonder window breaks?"
TOL = 1e-3


def ensure_zig_built() -> None:
    if not ZIG_BIN.exists():
        subprocess.run(["zig", "build"], cwd=ZIG_DIR, check=True)


def zig_logits(prompt: str) -> np.ndarray:
    """Run the Zig engine in `logits` mode and read back its raw-f32 dump."""
    out = MODULE / "python" / ".parity_zig_logits.bin"
    subprocess.run(
        [
            str(ZIG_BIN), "logits",
            "--prompt", prompt,
            "--artifacts", str(ARTIFACTS),
            "--out", str(out),
        ],
        check=True,
        capture_output=True,
    )
    data = np.frombuffer(out.read_bytes(), dtype="<f4").copy()
    out.unlink(missing_ok=True)
    return data


def ref_logits(prompt: str) -> np.ndarray:
    model, stoi = load_reference(ARTIFACTS)
    idx = encode(prompt, stoi)
    with torch.no_grad():
        logits = model(idx)          # (1, T, vocab)
    return logits[0, -1, :].numpy().astype(np.float32)


def make_figure(ref: np.ndarray, zig: np.ndarray, maxdiff: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diffs = zig - ref
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    lo, hi = float(min(ref.min(), zig.min())), float(max(ref.max(), zig.max()))
    ax1.plot([lo, hi], [lo, hi], color="#bbbbbb", lw=1, zorder=1, label="y = x")
    ax1.scatter(ref, zig, s=26, color="#2a6f97", alpha=0.85, zorder=2, edgecolor="white", linewidth=0.4)
    ax1.set_xlabel("PyTorch logit")
    ax1.set_ylabel("Zig logit")
    ax1.set_title("Per-token logits: Zig vs PyTorch")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, alpha=0.25)

    ax2.hist(diffs, bins=25, color="#e07a5f", edgecolor="white")
    ax2.axvline(0, color="#333", lw=1)
    ax2.set_xlabel("Zig − PyTorch  (logit units)")
    ax2.set_ylabel("count")
    ax2.set_title(f"Error histogram (max |Δ| = {maxdiff:.2e})")
    ax2.grid(True, alpha=0.25)

    fig.suptitle(f'Parity on "{PROMPT[:32]}…"  —  tolerance {TOL:g}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "parity_scatter.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_parity(make_fig: bool = True) -> float:
    ensure_zig_built()
    ref = ref_logits(PROMPT)
    zig = zig_logits(PROMPT)
    assert ref.shape == zig.shape, f"shape mismatch {ref.shape} vs {zig.shape}"
    maxdiff = float(np.max(np.abs(zig - ref)))
    if make_fig:
        path = make_figure(ref, zig, maxdiff)
        print(f"figure -> {path.relative_to(MODULE)}")
    return maxdiff


def main() -> None:
    maxdiff = run_parity(make_fig=True)
    print(f"prompt         : {PROMPT!r}")
    print(f"max |Δ logit|  : {maxdiff:.3e}   (tolerance {TOL:g})")
    if maxdiff < TOL:
        print("PARITY OK — the Zig engine matches PyTorch.")
    else:
        print("PARITY FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
