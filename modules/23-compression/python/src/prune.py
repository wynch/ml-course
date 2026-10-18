"""Section 2: magnitude pruning of SmolLM2-360M-Instruct at 10, 30 and 50%.

In every attention and MLP matrix of the 32 decoder blocks (7 per block, 224
in all), zero the given fraction of weights with the smallest |w|. Each matrix
loses the same fraction ("per matrix", not one global threshold). The tied
embedding, which is also the output layer, and the norms stay as they are.

A control zeroes the same number of weights per matrix at random positions
(torch.Generator seeded with SEED), so the table shows what choosing by
magnitude buys over choosing blindly.

Perplexity: the first 32 x 1024 tokens of the wikitext slice, 32,736
predicted tokens (see common.perplexity). The model runs in float32 on MPS.
Each setting also answers one chat request, 40 greedy tokens, to read by eye.

Run me:
    uv run python src/prune.py            # numbers, results/prune.json
    uv run python src/prune.py --figures  # also figures/prune.png
"""

from __future__ import annotations

import argparse

import common  # noqa: F401  (MPS guard before torch)
import torch

from common import (SEED, Guard, free, get_device, load_hf_model, load_tokenizer, perplexity, ppl_ids, seed_all,
                    write_results)
from compress import block_linears, magnitude_mask, random_mask

FRACTIONS = (0.1, 0.3, 0.5)
REQUEST = "Explain in two sentences why the sky is blue."
NEW_TOKENS = 40


@torch.no_grad()
def reply(model, tok) -> str:
    msgs = [{"role": "user", "content": REQUEST}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = {k: v.to(next(model.parameters()).device) for k, v in ids.items()}
    out = model.generate(**ids, max_new_tokens=NEW_TOKENS, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def mps_vs_cpu(model, ids: torch.Tensor) -> dict:
    """One forward pass of the first 256 tokens on the CPU and on MPS."""
    x = ids[:1, :256]
    cpu = model.to("cpu")(x).logits
    mps = model.to("mps")(x.to("mps")).logits.cpu()
    return {"tokens": 256, "max_abs_diff_logits": float((cpu - mps).abs().max()),
            "logit_range": [round(float(cpu.min()), 2), round(float(cpu.max()), 2)],
            "same_top1_every_position": bool((cpu.argmax(-1) == mps.argmax(-1)).all())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    guard = Guard("prune.py")
    seed_all()
    device = get_device()
    tok = load_tokenizer()
    ids = ppl_ids()
    model = load_hf_model("cpu")
    check = mps_vs_cpu(model, ids) if device.type == "mps" else None
    model.to(device)
    print(f"MPS vs CPU: {check}")
    guard.tick("load")

    mats = list(block_linears(model))
    n_block = sum(m.weight.numel() for _, m in mats)
    n_all = sum(p.numel() for p in model.parameters())
    # The checkpoint is bfloat16, so a bf16 copy restores the float32 weights exactly.
    originals = {n: m.weight.detach().to("cpu", torch.bfloat16).clone() for n, m in mats}

    def restore():
        for n, m in mats:
            m.weight.data.copy_(originals[n].float())

    base = perplexity(model, ids)
    rows = [{"method": "untouched", "fraction": 0.0, **base, "reply": reply(model, tok)}]
    print(f"untouched: ppl {base['ppl']}")
    guard.tick("untouched")

    gen = torch.Generator().manual_seed(SEED)
    for method in ("magnitude", "random"):
        for f in FRACTIONS:
            restore()
            for _, m in mats:
                mask = magnitude_mask(m.weight, f) if method == "magnitude" else random_mask(m.weight, f, gen)
                m.weight.data.mul_(mask)
            zeros = sum(int((m.weight == 0).sum()) for _, m in mats)
            r = perplexity(model, ids)
            rows.append({"method": method, "fraction": f, **r, "zero_share_of_block_weights": round(zeros / n_block, 4),
                         "zero_share_of_all_params": round(zeros / n_all, 4), "reply": reply(model, tok)})
            print(f"{method:9s} {f:.0%}: ppl {r['ppl']}  ({rows[-1]['reply'][:60]!r})")
            guard.tick(f"{method}_{f}")
    restore()

    out = {
        "model": common.MODEL_ID, "revision": common.MODEL_REVISION, "dtype": "float32", "device": str(device),
        "matrices": len(mats), "block_weights": n_block, "all_params": n_all,
        "ppl_setup": {"windows": common.PPL_WINDOWS, "window_tokens": common.PPL_LEN,
                      "predicted_tokens": base["tokens"], "text": "wikitext-103 train, first two row groups"},
        "request": REQUEST, "new_tokens": NEW_TOKENS, "seed": SEED,
        "rows": rows, "mps_vs_cpu": check,
    }
    del model, originals
    free()
    out["guard"] = guard.report()
    write_results("prune", out)
    if args.figures:
        import figures

        figures.prune()


if __name__ == "__main__":
    main()
