"""Anatomy of a real LLM: walk the module tree, count parameters, draw where
they live, and put SmolLM2 next to the tiny GPT from module 04 and SmolLM3-3B.

Run me:
    uv run python src/anatomy.py            # print the architecture summary
    uv run python src/anatomy.py --figure   # also (re)draw the treemap + table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import MODEL_ID, load_model, load_tokenizer

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


# ---------------------------------------------------------------------------
# Config-level param accounting (works for any Llama/SmolLM-style config).
# Same formula for the toy and the giants — only the numbers change.
# ---------------------------------------------------------------------------
def component_params(cfg: dict) -> dict[str, int]:
    """Return parameter counts per component from a config dict.

    Assumes a modern decoder: RMSNorm (weight only, no bias), no attention/MLP
    bias, SwiGLU MLP (gate+up+down), GQA (kv heads may be < query heads), and
    RoPE (no learned positional embeddings). Handles tied word embeddings.
    """
    V = cfg["vocab_size"]
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    n_head = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_head)
    head_dim = cfg.get("head_dim", H // n_head)
    inter = cfg["intermediate_size"]
    tied = cfg.get("tie_word_embeddings", True)

    q = H * (n_head * head_dim)
    k = H * (n_kv * head_dim)
    v = H * (n_kv * head_dim)
    o = (n_head * head_dim) * H
    attn = (q + k + v + o) * L

    mlp = (3 * H * inter) * L  # gate, up, down

    norms = (2 * H) * L + H  # 2 RMSNorm per layer + final norm

    embed = V * H
    head = 0 if tied else V * H

    total = attn + mlp + norms + embed + head
    return {
        "embeddings": embed,
        "attention": attn,
        "mlp": mlp,
        "norms": norms,
        "lm_head": head,
        "total": total,
    }


# The three columns of the comparison table.
# tiny-GPT: a *representative* char-level GPT of the kind you build in module 04
# (learned positional embeddings + LayerNorm; counted separately below). Its
# exact size depends on the hyper-parameters you pick there.
TINY_GPT = {
    "name": "tiny char-GPT (module 04)",
    "vocab_size": 65,
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 512,
    "max_position_embeddings": 256,
    "tie_word_embeddings": False,
    "positional": "learned",
}
SMOLLM2 = {
    "name": "SmolLM2-360M-Instruct",
    "vocab_size": 49152,
    "hidden_size": 960,
    "num_hidden_layers": 32,
    "num_attention_heads": 15,
    "num_key_value_heads": 5,
    "intermediate_size": 2560,
    "max_position_embeddings": 8192,
    "tie_word_embeddings": True,
    "positional": "RoPE",
}
SMOLLM3 = {  # from HuggingFaceTB/SmolLM3-3B config.json (weights NOT downloaded)
    "name": "SmolLM3-3B",
    "vocab_size": 128256,
    "hidden_size": 2048,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "intermediate_size": 11008,
    "max_position_embeddings": 65536,
    "tie_word_embeddings": True,
    "positional": "RoPE (NoPE every 4th layer)",
}


def tiny_gpt_params(cfg: dict) -> int:
    """Char-GPT param count: SwiGLU-free MLP + learned positional embeddings."""
    V, H, L = cfg["vocab_size"], cfg["hidden_size"], cfg["num_hidden_layers"]
    ctx, inter = cfg["max_position_embeddings"], cfg["intermediate_size"]
    embed = V * H + ctx * H  # token + learned positional
    attn = (4 * H * H) * L  # q,k,v,o (no GQA in the toy)
    mlp = (2 * H * inter) * L  # up + down (GELU, no gate)
    norms = (2 * H) * L + H
    head = V * H  # untied
    return embed + attn + mlp + norms + head


def human(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(n)


# ---------------------------------------------------------------------------
# Walk the *real* loaded model and print a summary grounded in actual modules.
# ---------------------------------------------------------------------------
def print_architecture_summary(model, tokenizer) -> dict[str, int]:
    cfg = model.config
    print(f"\n{'=' * 66}\n  {MODEL_ID}\n{'=' * 66}")
    print(f"  class            {model.__class__.__name__}")
    print(f"  hidden_size      {cfg.hidden_size}")
    print(f"  layers           {cfg.num_hidden_layers}")
    print(f"  attention heads  {cfg.num_attention_heads}  (head_dim {cfg.hidden_size // cfg.num_attention_heads})")
    print(f"  kv heads (GQA)   {cfg.num_key_value_heads}")
    print(f"  mlp intermediate {cfg.intermediate_size}")
    print(f"  vocab / context  {cfg.vocab_size} / {cfg.max_position_embeddings}")
    print(f"  tied embeddings  {cfg.tie_word_embeddings}")

    # Group the true parameter tensors by role.
    buckets = {"embeddings": 0, "attention": 0, "mlp": 0, "norms": 0, "lm_head": 0, "other": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        if "embed_tokens" in name:
            buckets["embeddings"] += n
        elif "self_attn" in name:
            buckets["attention"] += n
        elif "mlp" in name:
            buckets["mlp"] += n
        elif "norm" in name:
            buckets["norms"] += n
        elif "lm_head" in name:
            buckets["lm_head"] += n
        else:
            buckets["other"] += n

    total = sum(buckets.values())
    print(f"\n  parameters by component (real tensors, tied head shares embeddings):")
    for k, v in buckets.items():
        if v:
            print(f"    {k:<12} {human(v):>8}   {100 * v / total:5.1f}%")
    print(f"    {'-' * 28}")
    print(f"    {'total':<12} {human(total):>8}")

    # A single transformer block, unrolled — the thing you built in module 04.
    print(f"\n  one decoder block (x{cfg.num_hidden_layers}):")
    block = model.model.layers[0]
    for sub, mod in block.named_children():
        np = sum(p.numel() for p in mod.parameters())
        print(f"    {sub:<22} {mod.__class__.__name__:<26} {human(np):>8}")

    return buckets


# ---------------------------------------------------------------------------
# Figure 1: where the parameters live (stacked bar) + comparison table.
# ---------------------------------------------------------------------------
def draw_param_figure(buckets: dict[str, int], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = [(k, v) for k, v in buckets.items() if v and k != "other"]
    order = ["embeddings", "attention", "mlp", "norms", "lm_head"]
    parts.sort(key=lambda kv: order.index(kv[0]))
    labels = [k for k, _ in parts]
    values = [v for _, v in parts]
    total = sum(values)
    colors = {
        "embeddings": "#4C72B0",
        "attention": "#DD8452",
        "mlp": "#55A868",
        "norms": "#C44E52",
        "lm_head": "#8172B3",
    }

    fig, (ax_bar, ax_tab) = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.5]}
    )

    # --- stacked bar: SmolLM2 parameter budget ---
    bottom = 0
    for lbl, val in parts:
        ax_bar.bar(0, val, bottom=bottom, width=0.6, color=colors[lbl], edgecolor="white")
        if val / total > 0.03:
            ax_bar.text(
                0, bottom + val / 2, f"{lbl}\n{human(val)}  ({100 * val / total:.0f}%)",
                ha="center", va="center", color="white", fontsize=9, fontweight="bold",
            )
        bottom += val
    ax_bar.set_xlim(-0.6, 0.6)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("parameters")
    ax_bar.set_title(f"Where SmolLM2's {human(total)} parameters live", fontsize=11)
    ax_bar.yaxis.set_major_formatter(lambda x, _: human(int(x)))
    ax_bar.spines[["top", "right"]].set_visible(False)

    # --- comparison table: same blocks, bigger numbers ---
    tiny_total = tiny_gpt_params(TINY_GPT)
    sm2_total = component_params(SMOLLM2)["total"]
    sm3_total = component_params(SMOLLM3)["total"]
    ax_tab.axis("off")
    rows = [
        ("parameters", human(tiny_total), human(sm2_total), human(sm3_total)),
        ("hidden size", "128", "960", "2048"),
        ("layers", "4", "32", "36"),
        ("attn heads", "4", "15", "16"),
        ("kv heads (GQA)", "4", "5", "4"),
        ("mlp intermediate", "512", "2560", "11008"),
        ("vocab size", "65", "49,152", "128,256"),
        ("context length", "256", "8,192", "65,536"),
        ("positional", "learned", "RoPE", "RoPE+NoPE"),
    ]
    col_labels = ["", "tiny char-GPT\n(module 04)", "SmolLM2-360M", "SmolLM3-3B"]
    table = ax_tab.table(
        cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
        colColours=["#f0f0f0", "#e8edf5", "#e8f0ea", "#efe8f2"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(fontweight="bold", ha="left")
            cell.PAD = 0.03
    ax_tab.set_title("Same blocks you built in module 04 — bigger numbers", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true", help="also (re)draw the param figure")
    args = ap.parse_args()

    model = load_model()
    tok = load_tokenizer()
    buckets = print_architecture_summary(model, tok)

    print("\n  cross-model check (config formula vs bigger siblings):")
    for cfg in (TINY_GPT, SMOLLM2, SMOLLM3):
        if cfg is TINY_GPT:
            tot = tiny_gpt_params(cfg)
        else:
            tot = component_params(cfg)["total"]
        print(f"    {cfg['name']:<28} {human(tot):>8}")

    if args.figure:
        FIG_DIR.mkdir(exist_ok=True)
        draw_param_figure(buckets, FIG_DIR / "param_budget.png")


if __name__ == "__main__":
    main()
