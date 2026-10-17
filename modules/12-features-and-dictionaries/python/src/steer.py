"""Section 4: steering against prompting, a demo on SmolLM2-360M-Instruct.

A steering vector is a direction added to the residual stream while the model
generates. Ours is the plainest kind: the mean residual (output of
``model.model.layers[LAYER]``, averaged over tokens) on eight positive
sentences minus the same on eight negative ones. During generation a forward
hook adds a multiple of that vector at every position.

Five conditions, four neutral requests each, 48 greedy tokens:
  plain        the chat template's default system prompt, no edit
  prompt       system prompt "Reply in an upbeat tone."
  steer x2     plain, plus 2 x the sentiment vector at layer LAYER
  steer x4     plain, plus 4 x the sentiment vector
  random x4    plain, plus 4 x a random direction of the same norm (a control)

Score: P(positive) from distilbert fine-tuned on SST-2, loaded only after
SmolLM2 is freed. We tried layers 8, 12, 16 and 20 at x2, x4 and x8 on two of
these requests and picked by eye; this is a demo, not a measurement.

Run me:
    uv run python src/steer.py            # numbers, results/steer.json
    uv run python src/steer.py --figures  # also figures/steer.png
"""

from __future__ import annotations

import argparse

from common import (FIGURES, MODEL_ID, SENTIMENT_ID, Guard, free, get_device, load_hf_model,
                    load_tokenizer, write_results)

import numpy as np
import torch

LAYER = 16                 # model.model.layers[LAYER], counting from 0, of 32
SCALES = (2.0, 4.0)        # x2 reads fluently in our tries; x4 starts to break
MAX_NEW = 48
SEED = 0
POSITIVE = [
    "I love this. It made my whole day wonderful.",
    "What a fantastic result, I am so happy and proud.",
    "The food was delicious and the staff were lovely.",
    "This is the best news I have heard all year!",
    "We had a great time and laughed all evening.",
    "Thank you, this is beautiful and works perfectly.",
    "The film was brilliant, warm and full of joy.",
    "I feel calm, grateful and full of hope today.",
]
NEGATIVE = [
    "I hate this. It ruined my whole day.",
    "What a terrible result, I am so sad and ashamed.",
    "The food was disgusting and the staff were rude.",
    "This is the worst news I have heard all year.",
    "We had an awful time and argued all evening.",
    "No thanks, this is ugly and does not work at all.",
    "The film was dull, cold and full of misery.",
    "I feel tense, bitter and without hope today.",
]
REQUESTS = [
    "Describe a typical Monday morning.",
    "Tell me about the weather in autumn.",
    "What is a bus station like?",
    "Write two sentences about doing the laundry.",
]
UPBEAT = "Reply in an upbeat tone."


@torch.no_grad()
def mean_residual(model, tok, sentences: list[str], layer: int) -> torch.Tensor:
    """Mean over tokens, then over sentences, of layers[layer]'s output: (d_model,)."""
    out = []
    for s in sentences:
        ids = tok(s, return_tensors="pt").to(model.device)
        hs = model(**ids, output_hidden_states=True).hidden_states[layer + 1]   # [0] = embeddings
        out.append(hs[0].mean(0))
    return torch.stack(out).mean(0)


@torch.no_grad()
def typical_norm(model, tok, sentences: list[str], layer: int) -> float:
    norms = []
    for s in sentences:
        ids = tok(s, return_tensors="pt").to(model.device)
        hs = model(**ids, output_hidden_states=True).hidden_states[layer + 1]
        norms.append(hs[0, 1:].norm(dim=-1))
    return float(torch.cat(norms).median())


def chat(tok, request: str, system: str | None) -> torch.Tensor:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": request}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)


@torch.no_grad()
def generate(model, tok, request: str, system: str | None = None, vec: torch.Tensor | None = None,
             layer: int = LAYER) -> str:
    handle = None
    if vec is not None:
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + vec,) + tuple(output[1:])
            return output + vec
        handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        enc = chat(tok, request, system).to(model.device)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    finally:
        if handle:
            handle.remove()
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def p_positive(texts: list[str], device) -> list[float]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SENTIMENT_ID)
    clf = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_ID).to(device).eval()
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    p = clf(**enc).logits.softmax(-1)[:, clf.config.label2id["POSITIVE"]]
    return [round(float(v), 4) for v in p]


def run(figures: bool = False) -> dict:
    guard = Guard("steer.py")
    dev = get_device()
    model, tok = load_hf_model(), load_tokenizer()
    guard.tick("load")
    pos = mean_residual(model, tok, POSITIVE, LAYER)
    neg = mean_residual(model, tok, NEGATIVE, LAYER)
    vec = pos - neg
    g = torch.Generator().manual_seed(SEED)
    rnd = torch.randn(vec.shape, generator=g).to(vec.device)
    rnd = rnd / rnd.norm() * vec.norm()
    resid_norm = typical_norm(model, tok, REQUESTS, LAYER)
    conditions = {"plain": dict(), "prompt": dict(system=UPBEAT)}
    for sc in SCALES:
        conditions[f"steer x{sc:g}"] = dict(vec=sc * vec)
    conditions[f"random x{SCALES[-1]:g}"] = dict(vec=SCALES[-1] * rnd)
    replies = {name: [generate(model, tok, r, **kw) for r in REQUESTS] for name, kw in conditions.items()}
    for name, rs in replies.items():
        print(f"--- {name}")
        for r in rs:
            print("   ", r.replace("\n", " ")[:150])
    guard.tick("generate")
    del model
    free()

    names = list(replies)
    flat = [r for n in names for r in replies[n]]
    scores = p_positive(flat, dev)
    guard.tick("score")
    out_conds = {}
    for i, n in enumerate(names):
        s = scores[i * len(REQUESTS) : (i + 1) * len(REQUESTS)]
        out_conds[n] = {"replies": replies[n], "p_positive": s, "mean_p_positive": round(float(np.mean(s)), 3)}
        print(f"{n}: mean P(positive) {out_conds[n]['mean_p_positive']}  {s}")

    out = {
        "model": f"{MODEL_ID}, float32, greedy, {MAX_NEW} new tokens",
        "scorer": f"{SENTIMENT_ID}, P(POSITIVE)",
        "layer": LAYER,
        "scales": list(SCALES),
        "vector": "mean over tokens of layers[LAYER] output, positive sentences minus negative sentences",
        "vector_norm": round(float(vec.norm()), 3),
        "median_token_residual_norm_on_requests": round(resid_norm, 3),
        "residual_norm_note": "median over tokens of ||layers[LAYER] output||, first token left out "
                              "(it carries a far larger norm)",
        "positive_sentences": POSITIVE,
        "negative_sentences": NEGATIVE,
        "requests": REQUESTS,
        "system_prompt": UPBEAT,
        "conditions": out_conds,
        "note": "demo: 4 requests, one layer, scales chosen by eye; not a measurement",
    }
    if figures:
        fig_steer(out, FIGURES / "steer.png")
    guard.tick("figures")
    out["guard"] = guard.report()
    write_results("steer", out)
    return out


def fig_steer(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = res["conditions"]
    names = list(conds)
    def label(n):
        if n == "plain":
            return "plain"
        if n == "prompt":
            return "system prompt\n“upbeat”"
        if n.startswith("steer"):
            return f"steered, layer {res['layer']}\n{n.split()[1].replace('x', '×')}"
        return f"random direction\n{n.split()[1].replace('x', '×')}, same norm"
    labels = {n: label(n) for n in names}
    colors = {n: {"plain": "#9dada8", "prompt": "#453781"}.get(n, "#e07a5f" if n.startswith("random") else "#b8336a")
              for n in names}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, n in enumerate(names):
        s = conds[n]["p_positive"]
        ax.scatter([i] * len(s), s, color=colors[n], s=40, zorder=3)
        ax.hlines(conds[n]["mean_p_positive"], i - 0.25, i + 0.25, color=colors[n], lw=2.5)
        ax.text(i + 0.28, conds[n]["mean_p_positive"], f"{conds[n]['mean_p_positive']:.2f}", va="center", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([labels[n] for n in names], fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("P(positive), distilbert SST-2")
    ax.set_title("Four requests per condition (dots), mean (bar). A demo, not a measurement.", fontsize=10)
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
