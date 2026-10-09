"""Generate every figure the README embeds (except the parity scatter, which
scripts/parity.py owns):

  kv_cache_schematic.png   what the cache stores: K/V per layer/head, growing per step
  kv_cache_timing.png      tokens/sec vs sequence length, cache ON (linear) vs OFF (quadratic)
  quant_weight_hist.png    weight distribution + the int8 grid it snaps to
  quant_filesize.png       2.48 MB f32  ->  ~0.63 MB int8
  quant_parity_hist.png    int8-vs-f32 logit error histogram
  quant_tokens_per_sec.png f32 vs int8 decode speed

Run:  uv run python scripts/figures.py
(Reads zig/figures... no — reads the bench CSV produced by `tiny-gpt bench`;
 regenerates it if missing.)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODULE = Path(__file__).resolve().parents[2]
ARTIFACTS = MODULE.parent / "04-attention-transformer" / "artifacts"
ZIG_DIR = MODULE / "zig"
ZIG_BIN = ZIG_DIR / "zig-out" / "bin" / "tiny-gpt"
FIGURES = MODULE / "figures"
Q8 = ZIG_DIR / "tiny_gpt_weights.q8.bin"
BENCH_CSV = FIGURES / "bench.csv"

PROMPT = "ROMEO: But soft, what light through yonder window breaks?"

INK = "#2b2b2b"
BLUE = "#2a6f97"
RED = "#e07a5f"
GREEN = "#5a8f5a"
GREY = "#c8c8c8"


def sh(args, **kw):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kw)


def ensure_setup():
    if not ZIG_BIN.exists():
        sh(["zig", "build"], cwd=ZIG_DIR)
    if not Q8.exists():
        sh([str(ZIG_BIN), "quantize", "--artifacts", str(ARTIFACTS), "--out", str(Q8)], cwd=ZIG_DIR)
    if not BENCH_CSV.exists():
        sh([str(ZIG_BIN), "bench", "--artifacts", str(ARTIFACTS), "--out", str(BENCH_CSV)], cwd=ZIG_DIR)
    FIGURES.mkdir(exist_ok=True)


def load_config():
    return json.loads((ARTIFACTS / "tiny_gpt_config.json").read_text())


def load_tensor(name: str) -> np.ndarray:
    cfg = load_config()
    blob = (ARTIFACTS / "tiny_gpt_weights.bin").read_bytes()
    t = next(x for x in cfg["tensors"] if x["name"] == name)
    return np.frombuffer(blob[t["offset"]: t["offset"] + t["count"] * 4], dtype="<f4").reshape(t["shape"])


def zig_logits(prompt: str, q8: bool) -> np.ndarray:
    out = MODULE / "python" / ".fig_logits.bin"
    args = [str(ZIG_BIN), "logits", "--prompt", prompt, "--artifacts", str(ARTIFACTS), "--out", str(out)]
    if q8:
        args += ["--q8", str(Q8)]
    sh(args, cwd=ZIG_DIR)
    data = np.frombuffer(out.read_bytes(), dtype="<f4").copy()
    out.unlink(missing_ok=True)
    return data


def measure_tok_s(q8: bool, runs: int = 5) -> float:
    """Best-of-N decode tok/s from the Zig `generate --timing` line."""
    best = 0.0
    for _ in range(runs):
        args = [str(ZIG_BIN), "generate", "--prompt", "ROMEO:", "--tokens", "121",
                "--temperature", "0.0", "--artifacts", str(ARTIFACTS), "--timing"]
        if q8:
            args += ["--q8", str(Q8)]
        res = sh(args, cwd=ZIG_DIR)
        for line in res.stderr.splitlines():
            if line.startswith("TIMING"):
                tok_s = float(line.split("tok_s=")[1].split()[0])
                best = max(best, tok_s)
    return best


# --------------------------------------------------------------------------- #
# 1. KV-cache schematic
# --------------------------------------------------------------------------- #

def fig_cache_schematic():
    cfg = load_config()
    n_layer, n_head, d_head = cfg["n_layer"], cfg["n_head"], cfg["d_head"]
    steps = 8
    cw, chh, gap = 0.5, 0.5, 0.06

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})

    # --- left: the cache growing one column per decode step (a staircase) ---
    axL.axis("off")
    axL.set_title("The KV cache grows one column per decode step", fontsize=12, weight="bold", loc="left")
    for t in range(steps):
        y = (steps - 1 - t) * (chh + 0.10)
        axL.text(-0.35, y + chh / 2, f"step t={t}", ha="right", va="center", fontsize=9, color=INK)
        for pos in range(t + 1):
            x = pos * (cw + gap)
            newest = pos == t
            rect = plt.Rectangle((x, y), cw, chh,
                                 facecolor=RED if newest else BLUE,
                                 alpha=1.0 if newest else 0.55,
                                 edgecolor="white", lw=1.0)
            axL.add_patch(rect)
    axL.text(0, steps * (chh + 0.10) + 0.05,
             "each cell = one cached key K[pos]  (values V[pos] mirror it).  "
             "The red cell is this step's new token;\nblue cells are reused unchanged — that reuse is the whole point of the cache.",
             fontsize=8.6, color="#555", va="bottom")
    axL.set_xlim(-2.2, steps * (cw + gap) + 0.3)
    axL.set_ylim(-0.6, steps * (chh + 0.10) + 1.0)

    # --- right: the full cache tensor shape, per layer/head ---
    axR.axis("off")
    axR.set_title("What's actually stored", fontsize=12, weight="bold", loc="left")
    lines = [
        "For a sequence of T tokens:",
        "",
        f"K cache:  {n_layer} layers",
        f"          × {n_head} heads",
        f"          × T positions",
        f"          × {d_head} dims  (d_head)",
        "V cache:  the same, again",
        "",
        f"At T={cfg['block_size']} (full context) that is",
        f"2 × {n_layer} × {n_head} × {cfg['block_size']} × {d_head}",
        f"= {2 * n_layer * n_head * cfg['block_size'] * d_head:,} floats",
        f"≈ {2 * n_layer * n_head * cfg['block_size'] * d_head * 4 / 1024:.0f} KB.",
        "",
        "Real LLMs: this is why long",
        "contexts eat so much memory.",
    ]
    axR.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9.5,
             family="monospace", color=INK, transform=axR.transAxes)

    fig.tight_layout()
    p = FIGURES / "kv_cache_schematic.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("wrote", p.name)


# --------------------------------------------------------------------------- #
# 2. KV-cache timing curve
# --------------------------------------------------------------------------- #

def fig_cache_timing():
    rows = [l.split(",") for l in BENCH_CSV.read_text().splitlines()[1:]]
    seq = np.array([int(r[0]) for r in rows])
    on = np.array([float(r[1]) for r in rows])
    off = np.array([float(r[2]) for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.plot(seq, on, "-o", color=BLUE, label="KV cache ON (decode step is O(T))")
    ax1.plot(seq, off, "-s", color=RED, label="KV cache OFF (recompute, O(T²)/step)")
    ax1.set_xlabel("sequence length (tokens)")
    ax1.set_ylabel("throughput (tokens / sec)")
    ax1.set_title("Throughput vs context length")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.plot(seq, on / off, "-o", color=GREEN)
    ax2.set_xlabel("sequence length (tokens)")
    ax2.set_ylabel("speedup  (cache ON / OFF)")
    ax2.set_title("The cache pays off more as context grows")
    ax2.grid(True, alpha=0.25)
    for x, y in zip(seq, on / off):
        if x in (16, 64, 128):
            ax2.annotate(f"{y:.0f}×", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, color=GREEN)

    fig.suptitle("KV cache: linear vs quadratic decode", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIGURES / "kv_cache_timing.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("wrote", p.name)


# --------------------------------------------------------------------------- #
# 3. Quantization figures
# --------------------------------------------------------------------------- #

def fig_quant_weights():
    W = load_tensor("blocks.0.mlp.fc.weight")  # [512, 128]
    row = W[0]                                   # one output row -> one scale
    scale = np.max(np.abs(row)) / 127.0
    q = np.clip(np.round(row / scale), -127, 127)
    recon = q * scale

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # panel A: distribution of all weights in the tensor
    ax1.hist(W.ravel(), bins=80, color=BLUE, alpha=0.85, edgecolor="white")
    ax1.set_title("Weight distribution (blocks.0.mlp.fc.weight)")
    ax1.set_xlabel("weight value")
    ax1.set_ylabel("count")
    ax1.grid(True, alpha=0.25)
    ax1.text(0.02, 0.95, f"int8 step for this tensor's max row\n≈ {scale:.2e}",
             transform=ax1.transAxes, va="top", fontsize=9, color="#555")

    # panel B: a single row snapping onto the 8-bit grid
    idx = np.arange(40)  # first 40 weights of the row, for legibility
    levels = np.unique(np.round(row[idx] / scale)) * scale
    for lv in levels:
        ax2.axhline(lv, color=GREY, lw=0.6, zorder=1)
    ax2.scatter(idx, row[idx], s=30, color=BLUE, zorder=3, label="f32 weight")
    ax2.scatter(idx, recon[idx], s=18, color=RED, marker="D", zorder=4, label="int8 dequant")
    ax2.vlines(idx, row[idx], recon[idx], color="#aaa", lw=0.8, zorder=2)
    ax2.set_title("Values snap to the int8 grid")
    ax2.set_xlabel("weight index (within one row)")
    ax2.set_ylabel("value")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(False)

    fig.suptitle("int8 symmetric quantization (per row)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIGURES / "quant_weight_hist.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("wrote", p.name)


def fig_quant_filesize():
    f32_mb = (ARTIFACTS / "tiny_gpt_weights.bin").stat().st_size / 1e6
    q8_mb = Q8.stat().st_size / 1e6
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    bars = ax.bar(["f32\n(.bin)", "int8\n(Q8)"], [f32_mb, q8_mb], color=[BLUE, RED], width=0.6)
    ax.set_ylabel("file size (MB)")
    ax.set_title(f"Weight file: {f32_mb:.2f} MB → {q8_mb:.2f} MB  ({f32_mb / q8_mb:.1f}× smaller)")
    for b, v in zip(bars, [f32_mb, q8_mb]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f} MB", ha="center", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, f32_mb * 1.15)
    fig.tight_layout()
    p = FIGURES / "quant_filesize.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("wrote", p.name)


def fig_quant_parity():
    ref = zig_logits(PROMPT, q8=False)
    q8 = zig_logits(PROMPT, q8=True)
    diffs = q8 - ref
    maxdiff = float(np.max(np.abs(diffs)))
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.hist(diffs, bins=25, color=RED, edgecolor="white")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("int8 logit − f32 logit")
    ax.set_ylabel("count")
    ax.set_title(f"int8 vs f32 logits  (max |Δ| = {maxdiff:.3f})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = FIGURES / "quant_parity_hist.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("wrote", p.name, f"(int8 max|Δ|={maxdiff:.3f})")
    return maxdiff


def fig_quant_speed():
    f32 = measure_tok_s(q8=False)
    q8 = measure_tok_s(q8=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    bars = ax.bar(["f32", "int8\n(dequant→f32)"], [f32, q8], color=[BLUE, RED], width=0.6)
    ax.set_ylabel("decode throughput (tokens / sec)")
    ax.set_title("Decode speed: f32 vs int8")
    for b, v in zip(bars, [f32, q8]):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(f32, q8) * 1.2)
    fig.tight_layout()
    p = FIGURES / "quant_tokens_per_sec.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("wrote", p.name, f"(f32={f32:.0f}, int8={q8:.0f} tok/s)")


def main():
    ensure_setup()
    fig_cache_schematic()
    fig_cache_timing()
    fig_quant_weights()
    fig_quant_filesize()
    fig_quant_parity()
    fig_quant_speed()
    print("all figures ->", FIGURES.relative_to(MODULE))


if __name__ == "__main__":
    main()
