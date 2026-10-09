"""Exercise C harness (PROVIDED — you don't edit this file).

Once you've implemented the perplexity plumbing (ex_c_perplexity.zig, and the
engine's `perplexity` subcommand it mirrors), this harness runs the full engine
over a held-out Shakespeare slice with the f32 weights and again with the int8
Q8 weights, and reports both perplexities plus the gap.

The question it answers: *how much accuracy does 4x-smaller int8 actually cost?*

Run (from python/):  uv run python ../exercises/ex_c_perplexity.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
ARTIFACTS = MODULE.parent / "04-attention-transformer" / "artifacts"
ZIG_DIR = MODULE / "zig"
ZIG_BIN = ZIG_DIR / "zig-out" / "bin" / "tiny-gpt"
Q8 = ZIG_DIR / "tiny_gpt_weights.q8.bin"
HELDOUT = MODULE / "corpus" / "heldout.txt"


def ensure_ready() -> None:
    if not ZIG_BIN.exists():
        subprocess.run(["zig", "build"], cwd=ZIG_DIR, check=True)
    if not Q8.exists():
        subprocess.run(
            [str(ZIG_BIN), "quantize", "--artifacts", str(ARTIFACTS), "--out", str(Q8)],
            cwd=ZIG_DIR, check=True,
        )


def perplexity(q8: bool) -> float:
    args = [str(ZIG_BIN), "perplexity", "--artifacts", str(ARTIFACTS), "--text", str(HELDOUT)]
    if q8:
        args += ["--q8", str(Q8)]
    res = subprocess.run(args, cwd=ZIG_DIR, check=True, capture_output=True, text=True)
    line = next(l for l in res.stdout.splitlines() if l.startswith("PERPLEXITY"))
    return float(line.split("perplexity=")[1])


def main() -> None:
    ensure_ready()
    ppl_f32 = perplexity(q8=False)
    ppl_int8 = perplexity(q8=True)
    print(f"held-out slice : {HELDOUT.relative_to(MODULE)}  ({len(HELDOUT.read_text())} chars)")
    print(f"perplexity f32 : {ppl_f32:.4f}")
    print(f"perplexity int8: {ppl_int8:.4f}")
    print(f"cost of int8   : {ppl_int8 - ppl_f32:+.4f}  ({100 * (ppl_int8 - ppl_f32) / ppl_f32:+.2f}%)")
    print(
        "\nTakeaway: per-row int8 is nearly free in quality here — the whole model\n"
        "is 4x smaller on disk for a perplexity change in the third decimal place."
    )


if __name__ == "__main__":
    main()
