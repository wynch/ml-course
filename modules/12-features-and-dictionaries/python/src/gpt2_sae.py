"""Section 3: a real pretrained SAE, the library way.

GPT-2 small through TransformerLens 3.9, and Joseph Bloom's residual-stream
SAE for layer 8 (release ``gpt2-small-res-jb``, hook ``blocks.8.hook_resid_pre``,
24,576 latents) through SAELens 6. These SAEs are the older kind: a ReLU
encoder trained with an L1 penalty, not TopK.

What it does:
  1. checks one forward pass on MPS against the CPU;
  2. runs 512 rows of 128 tokens of wikitext-103 (BOS + 127 tokens each)
     through GPT-2, encodes the layer-8 residual with the SAE, and measures
     L0, variance explained, and the next-token loss when the SAE's
     reconstruction replaces the residual (BOS position left alone);
  3. keeps the top 10 contexts of every latent over those 65,024 tokens;
  4. for four prompts, lists the five latents with the largest activation,
     each with its Neuronpedia page.

Run me:
    uv run python src/gpt2_sae.py            # numbers, results/gpt2_sae.json
    uv run python src/gpt2_sae.py --figures  # also figures/gpt2_sae.png
"""

from __future__ import annotations

import argparse

from common import FIGURES, Guard, free, get_device, write_results

import numpy as np
import torch

from corpus import N_GROUPS, WIKITEXT_FILE, WIKITEXT_ID, WIKITEXT_REVISION, fetch_text

RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"
HOOK = "blocks.8.hook_resid_pre"
NEURONPEDIA = "https://www.neuronpedia.org/gpt2-small/8-res-jb/{}"
N_ROWS, SEQ = 512, 128
BATCH = 16
TOP = 10
BEFORE, AFTER = 24, 4
PROMPTS = [
    "The Eiffel Tower is located in the city of Paris.",
    "def add(a, b):\n    return a + b",
    "She was so happy that she started to cry.",
    "In 1912 the Titanic sank in the North Atlantic.",
]
TOP_PER_PROMPT = 5


def rows(model, text: str) -> torch.Tensor:
    ids = model.tokenizer(text, verbose=False).input_ids
    body = SEQ - 1
    need = N_ROWS * body
    assert len(ids) >= need, (len(ids), need)
    t = torch.tensor(ids[:need]).view(N_ROWS, body)
    bos = torch.full((N_ROWS, 1), model.tokenizer.bos_token_id)
    return torch.cat([bos, t], 1)


@torch.no_grad()
def mps_vs_cpu(model, dev) -> dict:
    toks = model.to_tokens(PROMPTS[0])
    a = model(toks.to(dev)).float().cpu()
    model.to("cpu")
    b = model(toks)
    model.to(dev)
    return {"prompt": PROMPTS[0], "max_abs_diff_logits": float((a - b).abs().max()),
            "logit_range": [round(float(b.min()), 2), round(float(b.max()), 2)],
            "same_top1_every_position": bool((a.argmax(-1) == b.argmax(-1)).all())}


@torch.no_grad()
def run(figures: bool = False) -> dict:
    guard = Guard("gpt2_sae.py")
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    dev = get_device()
    sae = SAE.from_pretrained(RELEASE, SAE_ID, device=str(dev))
    sae.eval()
    # Load GPT-2 the way the SAE was trained on it: no weight processing except
    # what its metadata asks for (here center_writing_weights=True).
    kwargs = sae.cfg.metadata.model_from_pretrained_kwargs or {}
    model = HookedTransformer.from_pretrained_no_processing("gpt2", device=str(dev), **kwargs)
    model.eval()
    d_sae = sae.cfg.d_sae
    guard.tick("load")
    check = mps_vs_cpu(model, dev)
    print(f"MPS vs CPU: max abs diff {check['max_abs_diff_logits']:.2e}, same top-1 {check['same_top1_every_position']}")

    text, info = fetch_text()
    toks = rows(model, text)
    n_tok = 0
    l0_sum, se, var_parts = 0.0, 0.0, []
    loss_intact, loss_splice, loss_zero = 0.0, 0.0, 0.0
    fires = torch.zeros(d_sae)
    top_v = torch.full((TOP, d_sae), -1.0)
    top_i = torch.zeros((TOP, d_sae), dtype=torch.long)
    resid_sum = None
    all_x = []

    for s in range(0, N_ROWS, BATCH):
        t = toks[s : s + BATCH].to(dev)
        logits, cache = model.run_with_cache(t, names_filter=HOOK)
        x = cache[HOOK]                                      # (B, T, 768)
        f = sae.encode(x)                                    # (B, T, d_sae)
        xh = sae.decode(f)
        body = (slice(None), slice(1, None))                 # skip BOS
        xb, fb, xhb = x[body], f[body], xh[body]
        n = xb.shape[0] * xb.shape[1]
        n_tok += n
        l0_sum += float((fb > 0).sum())
        se += float(((xb - xhb) ** 2).sum())
        all_x.append(xb.reshape(-1, xb.shape[-1]).cpu())
        fires += (fb > 0).sum((0, 1)).cpu()
        flat = fb.reshape(-1, d_sae)
        k = min(TOP, flat.shape[0])
        v, i = flat.topk(k, dim=0)
        # flat index -> (row, position) in toks; positions start at 1
        rows_i = s + i // (SEQ - 1)
        pos_i = i % (SEQ - 1) + 1
        gidx = (rows_i * SEQ + pos_i).cpu()
        allv = torch.cat([top_v, v.cpu()])
        alli = torch.cat([top_i, gidx])
        keep = allv.topk(TOP, dim=0).indices
        top_v, top_i = allv.gather(0, keep), alli.gather(0, keep)

        def ce(lg):
            lp = torch.log_softmax(lg[:, 1:-1].float(), -1)   # predict t+1 from positions 1..T-2
            tgt = t[:, 2:]
            return float(-lp.gather(-1, tgt[..., None]).sum())
        loss_intact += ce(logits)

        def splice_hook(act, hook):
            act = act.clone()
            act[:, 1:] = xh[:, 1:]
            return act

        def zero_hook(act, hook):
            act = act.clone()
            act[:, 1:] = 0.0
            return act

        loss_splice += ce(model.run_with_hooks(t, fwd_hooks=[(HOOK, splice_hook)]))
        loss_zero += ce(model.run_with_hooks(t, fwd_hooks=[(HOOK, zero_hook)]))
        del cache, f, xh, logits
        guard.tick()
    guard.tick("corpus")

    X = torch.cat(all_x)
    var = float(((X - X.mean(0)) ** 2).sum())
    n_pred = N_ROWS * (SEQ - 2)
    li, ls, lz = loss_intact / n_pred, loss_splice / n_pred, loss_zero / n_pred
    stats = {
        "tokens": n_tok,
        "l0": round(l0_sum / n_tok, 1),
        "fve": round(1 - se / var, 4),
        "loss_intact": round(li, 4),
        "loss_spliced": round(ls, 4),
        "loss_zero_ablated": round(lz, 4),
        "loss_recovered_vs_zero": round((lz - ls) / (lz - li), 4),
        "latents_firing_at_least_once": int((fires > 0).sum()),
        "d_sae": d_sae,
    }
    print(stats)
    flat_toks = toks.reshape(-1)

    def context(gidx: int) -> dict:
        r, p = divmod(int(gidx), SEQ)
        a, b = max(1, p - BEFORE), min(SEQ, p + AFTER + 1)
        return {"tokens": [model.tokenizer.decode([int(x)]) for x in toks[r, a:b]], "fire": p - a}

    def feature(idx: int) -> dict:
        vals = top_v[:, idx]
        ctx = []
        vmax = float(vals.max())
        for v, g in zip(vals.tolist(), top_i[:, idx].tolist()):
            if v <= 0:
                break
            ctx.append({**context(g), "activation": round(v, 3), "share_of_max": round(v / vmax, 3)})
        return {"index": idx, "url": NEURONPEDIA.format(idx), "freq": round(float(fires[idx]) / n_tok, 5),
                "corpus_max": round(vmax, 3) if ctx else 0.0, "contexts": ctx}

    prompts = []
    features: dict[int, dict] = {}
    for p in PROMPTS:
        t = model.to_tokens(p).to(dev)
        _, cache = model.run_with_cache(t, names_filter=HOOK)
        f = sae.encode(cache[HOOK])[0, 1:]                   # (T-1, d_sae), no BOS
        mx, where = f.max(0)
        v, idx = mx.topk(TOP_PER_PROMPT)
        strs = model.to_str_tokens(t)[1:]
        top = []
        for val, i in zip(v.tolist(), idx.tolist()):
            pos = int(where[i])
            top.append({"index": i, "activation": round(val, 3), "token": strs[pos], "position": pos,
                        "acts_by_token": [round(float(a), 3) for a in f[:, i]], "url": NEURONPEDIA.format(i)})
            if i not in features:
                features[i] = feature(i)
        prompts.append({"prompt": p, "tokens": strs, "top": top})
        print(f"{p!r}: " + ", ".join(f"{d['index']}@{d['token']!r} {d['activation']:.1f}" for d in top))
    guard.tick("prompts")

    out = {
        "model": "gpt2 (GPT-2 small, 124M) via TransformerLens HookedTransformer.from_pretrained_no_processing, float32",
        "model_kwargs": kwargs,
        "sae": {"release": RELEASE, "sae_id": SAE_ID, "d_sae": d_sae, "d_in": sae.cfg.d_in,
                "architecture": sae.cfg.architecture() if callable(getattr(sae.cfg, "architecture", None)) else str(getattr(sae.cfg, "architecture", "")),
                "neuronpedia": "gpt2-small/8-res-jb"},
        "versions": {"sae_lens": __import__("sae_lens").__version__,
                     "transformer_lens": __import__("importlib.metadata").metadata.version("transformer-lens")},
        "corpus": {"source": f"{WIKITEXT_ID}@{WIKITEXT_REVISION[:7]} {WIKITEXT_FILE}, first {N_GROUPS} row groups",
                   **info, "rows": N_ROWS, "tokens_per_row": SEQ, "note": "BOS + 127 tokens per row; BOS excluded"},
        "mps_vs_cpu": check,
        "stats": stats,
        "prompts": prompts,
        "features": [features[i] for i in features],
    }
    del model, sae
    free()
    if figures:
        fig_gpt2(out, FIGURES / "gpt2_sae.png")
    guard.tick("figures")
    out["guard"] = guard.report()
    write_results("gpt2_sae", out)
    return out


def fig_gpt2(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = res["prompts"]
    fig, axes = plt.subplots(len(ps), 1, figsize=(12, 2.3 * len(ps)))
    for ax, p in zip(axes, ps):
        toks = [t.replace("\n", "⏎") for t in p["tokens"]]
        M = np.array([d["acts_by_token"] for d in p["top"]])
        ax.imshow(M, aspect="auto", cmap="Reds", vmin=0)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, fontsize=8, rotation=40, ha="right")
        ax.set_yticks(range(len(p["top"])))
        ax.set_yticklabels([f"#{d['index']}" for d in p["top"]], fontsize=8)
        for r, d in enumerate(p["top"]):
            ax.text(d["position"], r, f"{d['activation']:.0f}", ha="center", va="center", fontsize=7)
    fig.suptitle("GPT-2 small, layer 8 residual: the five strongest SAE latents per prompt "
                 "(cell = activation; number = the latent's peak)", fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    run(figures=args.figures)


if __name__ == "__main__":
    main()
