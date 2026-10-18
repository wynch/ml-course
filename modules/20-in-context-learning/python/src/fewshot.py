"""Section 4, the library way: few-shot sentiment with SmolLM2-360M-Instruct on SST-2.

    uv run python src/fewshot.py [--figures]

Fixed before the first run:
- Model: HuggingFaceTB/SmolLM2-360M-Instruct at a pinned revision, float32, plain text,
  no chat template.
- Prompt: one instruction line, then k solved examples, then the test review:

      Classify the sentiment of each movie review as positive or negative.

      Review: <example>
      Sentiment: positive

      ...
      Review: <test review>
      Sentiment:

- Score: the next-token logits of " positive" and " negative" (one token each); the
  larger one is the answer. Nothing is generated.
- Test set: the SST-2 validation split, all 872 labelled sentences.
- Examples: drawn from SST-2 train sentences of 6 to 40 words. For each test sentence,
  numpy seed 0 draws 4 positive and 4 negative examples and shuffles their order; the
  k-shot prompt uses the first k of those 8, so k = 1, 4, 8 share examples.
- Conditions: k = 0, 1, 4, 8; and k = 8 with each example's label replaced by a fair
  coin (numpy seed 1), the shuffled-label control.
- Reported: accuracy with a 95% Wilson interval, and the share answered "positive".
"""

from __future__ import annotations

import argparse
import math

from common import Guard, free, get_device, load_hf_model, load_sst2, load_tokenizer, write_results

import numpy as np
import torch

HEADER = "Classify the sentiment of each movie review as positive or negative.\n\n"
LABELS = {0: "negative", 1: "positive"}
KS = (0, 1, 4, 8)
MAX_K = 8
WORDS = (6, 40)
SEED_DEMOS, SEED_RANDOM_LABELS = 0, 1
BATCH = 16


def format_prompt(test: str, demos: list[tuple[str, int]]) -> str:
    """The instruction, k solved examples, then the test review with its label left open."""
    shots = "".join(f"Review: {s}\nSentiment: {LABELS[y]}\n\n" for s, y in demos)
    return f"{HEADER}{shots}Review: {test}\nSentiment:"


def label_token_ids(tok) -> tuple[int, int]:
    """Token ids of " negative" and " positive"; each must be a single token."""
    neg, pos = tok.encode(" negative"), tok.encode(" positive")
    assert len(neg) == 1 and len(pos) == 1, (neg, pos)
    return neg[0], pos[0]


def draw_demos(train_s: list[str], train_y: list[int], n_test: int) -> list[list[tuple[str, int]]]:
    pool = {c: [i for i, (s, y) in enumerate(zip(train_s, train_y))
                if y == c and WORDS[0] <= len(s.split()) <= WORDS[1]] for c in (0, 1)}
    rng = np.random.default_rng(SEED_DEMOS)
    out = []
    for _ in range(n_test):
        idx = list(rng.choice(pool[1], MAX_K // 2, replace=False)) + list(rng.choice(pool[0], MAX_K // 2, replace=False))
        idx = [idx[j] for j in rng.permutation(MAX_K)]
        out.append([(train_s[i], train_y[i]) for i in idx])
    return out


@torch.no_grad()
def score(model, tok, prompts: list[str], ids: tuple[int, int]) -> np.ndarray:
    """logit(" positive") - logit(" negative") at the end of each prompt."""
    dev = model.device
    tok.padding_side = "left"
    order = np.argsort([len(p) for p in prompts])        # similar lengths share a batch: less padding
    out = np.zeros(len(prompts))
    for s in range(0, len(prompts), BATCH):
        sel = order[s:s + BATCH]
        enc = tok([prompts[i] for i in sel], return_tensors="pt", padding=True)
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids=enc.input_ids.to(dev), attention_mask=enc.attention_mask.to(dev),
                       position_ids=pos.to(dev), logits_to_keep=1).logits[:, -1].float().cpu().numpy()
        out[sel] = logits[:, ids[1]] - logits[:, ids[0]]
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("fewshot")

    test_s, test_y = load_sst2("validation")
    train_s, train_y = load_sst2("train")
    demos = draw_demos(train_s, train_y, len(test_s))
    rng = np.random.default_rng(SEED_RANDOM_LABELS)
    random_demos = [[(s, int(rng.integers(2))) for s, _ in d] for d in demos]
    conditions = {f"k={k}": [format_prompt(t, d[:k]) for t, d in zip(test_s, demos)] for k in KS}
    conditions["k=8, random labels"] = [format_prompt(t, d) for t, d in zip(test_s, random_demos)]
    guard.tick("data")

    tok = load_tokenizer()
    ids = label_token_ids(tok)
    model = load_hf_model()
    guard.tick("load")
    y = np.array(test_y)
    res = {}
    margins = {}
    for name, prompts in conditions.items():
        m = score(model, tok, prompts, ids)
        guard.tick(name)
        pred = (m > 0).astype(int)
        correct = int((pred == y).sum())
        res[name] = {"accuracy": correct / len(y), "correct": correct, "n": len(y), "ci95": wilson(correct, len(y)),
                     "share_positive": float(pred.mean()),
                     "mean_prompt_tokens": float(np.mean([len(tok.encode(p)) for p in prompts]))}
        margins[name] = m
        print(f"  {name}: accuracy {res[name]['accuracy']:.3f} {res[name]['ci95']}, "
              f"says positive {res[name]['share_positive']:.2f}")

    def paired(a: str, b: str) -> dict:
        ra, rb = (margins[a] > 0) == y, (margins[b] > 0) == y
        return {"a": a, "b": b, "only_a_right": int((ra & ~rb).sum()), "only_b_right": int((~ra & rb).sum())}

    pairs = [paired("k=8", "k=0"), paired("k=4", "k=0"), paired("k=8", "k=4"), paired("k=8", "k=8, random labels")]

    # The same forward on the CPU, for the first k = 8 prompt.
    p0 = conditions["k=8"][0]
    enc = tok(p0, return_tensors="pt")
    with torch.no_grad():
        l_dev = model(**{k: v.to(model.device) for k, v in enc.items()}).logits[0, -1].float().cpu()
    del model
    free()
    cpu_model = load_hf_model("cpu")
    with torch.no_grad():
        l_cpu = cpu_model(**enc).logits[0, -1].float()
    mps_vs_cpu = {"device": str(get_device()), "max_abs_diff_last_logits": float((l_dev - l_cpu).abs().max()),
                  "logit_range": [float(l_cpu.min()), float(l_cpu.max())],
                  "same_answer": bool((l_dev[ids[1]] > l_dev[ids[0]]) == (l_cpu[ids[1]] > l_cpu[ids[0]]))}
    del cpu_model
    free()

    examples = [{"test": test_s[i], "label": int(test_y[i]), "demos": [[s, l] for s, l in demos[i]],
                 "margin": {k: float(margins[k][i]) for k in margins}} for i in range(6)]
    write_results("fewshot", {
        "config": {"model_revision": "pinned in common.py", "header": HEADER, "ks": list(KS), "words": list(WORDS),
                   "label_token_ids": {"negative": ids[0], "positive": ids[1]}, "batch": BATCH,
                   "majority_share": float(max(y.mean(), 1 - y.mean())), "n_test": len(y)},
        "conditions": res,
        "paired": pairs,
        "mps_vs_cpu": mps_vs_cpu,
        "example_prompt_k1": conditions["k=1"][0],
        "examples": examples,
        "guard": guard.report(),
    })
    if args.figures:
        import figures
        figures.fig_fewshot()


if __name__ == "__main__":
    main()
