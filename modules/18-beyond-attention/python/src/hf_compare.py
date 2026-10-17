"""Section 4: Mamba-130M against GPT-2 small, the library way.

Both load with transformers' Auto classes and generate with ``generate``.
Mamba's fast path needs CUDA kernels (mamba-ssm, causal-conv1d); without
them transformers falls back to plain PyTorch, which runs the selective scan
as a Python loop over tokens. That is the path here, on Apple's GPU (MPS).

So the speed numbers below measure that fallback loop on this Mac, not
Mamba: on a CUDA GPU with the kernels, the scan runs fused. The memory
numbers measure the cache each model keeps between tokens, which does not
depend on the kernels.

Measured, per model:
- MPS against CPU: one forward pass on the same prompt, max |logit difference|
  and whether the top-1 token agrees at every position;
- one greedy reply, 48 new tokens, to a short prompt;
- for a context of 256 and of 960 tokens (GPT-2 stops at 1,024 positions):
  the bytes in the cache after the context, and new tokens per second for 32
  greedy tokens after it.

Run me:
    uv run python src/hf_compare.py            # numbers, results/hf_compare.json
    uv run python src/hf_compare.py --figures  # also figures/hf_compare.png
"""

from __future__ import annotations

import argparse
import time

from common import FIGURES, CORPUS, Guard, free, get_device, sync, write_results  # MPS guard before torch

import torch

MODELS = {
    # Pinned so the numbers in results/ can be reproduced exactly.
    "gpt2": {"id": "gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"},
    "mamba": {"id": "state-spaces/mamba-130m-hf", "revision": "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"},
}
PROMPT = "The old castle on the hill was"
NEW_TOKENS = 48
CONTEXTS = (256, 960)
DECODE = 32


def load(name: str, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = MODELS[name]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    model = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.float32)
    return model.to(device).eval(), tok


def cache_bytes(cache) -> int:
    """Bytes of every tensor the cache object holds, each counted once."""
    seen, total = set(), 0

    def visit(x, depth=0):
        nonlocal total
        if depth > 4:
            return
        if torch.is_tensor(x):
            if x.data_ptr() not in seen:
                seen.add(x.data_ptr())
                total += x.numel() * x.element_size()
        elif isinstance(x, (list, tuple)):
            for y in x:
                visit(y, depth + 1)
        elif isinstance(x, dict):
            for y in x.values():
                visit(y, depth + 1)
        elif hasattr(x, "__dict__"):
            for y in vars(x).values():
                visit(y, depth + 1)

    visit(cache)
    return total


def gen(model, ids, n, device):
    """Greedy generation of n tokens. Returns (output ids, seconds, cache)."""
    sync(device)
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                         return_dict_in_generate=True, pad_token_id=0)
    sync(device)
    cache = getattr(out, "past_key_values", None) or getattr(out, "cache_params", None)
    return out.sequences, time.perf_counter() - t0, cache


@torch.no_grad()
def measure(name: str, device, guard: Guard) -> dict:
    model, tok = load(name, device)
    n_params = sum(p.numel() for p in model.parameters())
    guard.tick(f"{name} load")

    # MPS against CPU, one forward pass
    ids = tok(PROMPT + " a place where", return_tensors="pt").input_ids
    a = model(ids.to(device)).logits.float().cpu()
    model.to("cpu")
    b = model(ids).logits.float()
    model.to(device)
    check = {"device": str(device), "tokens": ids.shape[1],
             "max_abs_diff_logits": float((a - b).abs().max()),
             "logit_range": [round(float(b.min()), 1), round(float(b.max()), 1)],
             "same_top1_every_position": bool((a.argmax(-1) == b.argmax(-1)).all())}

    # one greedy reply, after a short warm-up call (the first call on MPS pays one-off setup)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    gen(model, ids, 4, device)
    seq, secs, _ = gen(model, ids, NEW_TOKENS, device)
    reply = tok.decode(seq[0, ids.shape[1]:])

    # long contexts: cache size and decode speed
    text = CORPUS.read_text()[-60000:]
    all_ids = tok(text, return_tensors="pt").input_ids
    ctx = {}
    for T in CONTEXTS:
        ids = all_ids[:, :T].to(device)
        _, t1, cache = gen(model, ids, 1, device)              # the context plus one token
        nb = cache_bytes(cache)
        _, tn, _ = gen(model, ids, DECODE + 1, device)
        ctx[str(T)] = {"cache_bytes_after_context": nb,
                       "prefill_seconds": round(t1, 2),
                       "new_tokens_per_s": round(DECODE / max(tn - t1, 1e-9), 1)}
        guard.tick(f"{name} ctx {T}")
        print(name, T, ctx[str(T)])
    cfg = model.config.to_dict()
    del model
    free()
    return {"id": MODELS[name]["id"], "revision": MODELS[name]["revision"], "params": n_params,
            "config": {k: cfg.get(k) for k in ("n_layer", "n_embd", "n_head", "num_hidden_layers", "hidden_size",
                                                "intermediate_size", "state_size", "conv_kernel", "vocab_size")},
            "mps_vs_cpu": check, "prompt": PROMPT, "reply": reply,
            "reply_seconds": round(secs, 2), "reply_tokens_per_s": round(NEW_TOKENS / secs, 1),
            "contexts": ctx}


def run() -> dict:
    import transformers

    guard = Guard("hf_compare")
    device = get_device()
    out = {"device": str(device), "transformers": transformers.__version__, "models": {}}
    try:
        import mamba_ssm  # noqa: F401
        out["mamba_cuda_kernels"] = True
    except ImportError:
        out["mamba_cuda_kernels"] = False
    for name in ("gpt2", "mamba"):
        out["models"][name] = measure(name, device, guard)
        print(name, out["models"][name]["reply_tokens_per_s"], "tokens/s;", out["models"][name]["reply"][:80])
    out["guard"] = guard.report()
    return out


def fig_hf(r: dict, path) -> None:
    import matplotlib.pyplot as plt

    colors = {"gpt2": "#453781", "mamba": "#b7410e"}
    labels = {"gpt2": "GPT-2 small, KV cache", "mamba": "Mamba-130M, state"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    x = range(len(CONTEXTS))
    w = 0.38
    for i, name in enumerate(("gpt2", "mamba")):
        c = r["models"][name]["contexts"]
        axes[0].bar([j + (i - 0.5) * w for j in x], [c[str(T)]["cache_bytes_after_context"] / 2**20 for T in CONTEXTS],
                    w, color=colors[name], label=labels[name])
        axes[1].bar([j + (i - 0.5) * w for j in x], [c[str(T)]["new_tokens_per_s"] for T in CONTEXTS],
                    w, color=colors[name], label=labels[name])
    for ax, t in zip(axes, ("Cache after the context (MB)", f"New tokens per second, {r['device'].upper()}, pure-PyTorch path")):
        ax.set_xticks(list(x), [f"{T} tokens" for T in CONTEXTS])
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    r = run()
    write_results("hf_compare", r)
    if args.figures:
        fig_hf(r, FIGURES / "hf_compare.png")


if __name__ == "__main__":
    main()
