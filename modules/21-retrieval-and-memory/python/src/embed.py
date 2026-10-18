"""Section 1: two ways to turn text into one vector, compared on SQuAD paragraphs.

- **MiniLM**: all-MiniLM-L6-v2, a 22.7M-parameter BERT trained on a billion
  sentence pairs to put a question near its answer. Its recipe (the repo's
  modules.json): mean of the last hidden states over real tokens, then L2
  normalise. Inputs are cut at 256 tokens, as its sentence_bert_config says.
- **SmolLM2 mean-pool**: SmolLM2-360M-Instruct, never trained for this. The
  same recipe on its last hidden states (after the final norm), cut at 512.

Every one of the 2,067 validation paragraphs and 10,570 questions is
embedded by both; the vectors go to the Hugging Face cache folder (about 40
MB), where src/retrieval.py reads them. This script reports what the
vectors look like: cosine between a question and its own paragraph against a
random paragraph, and whether a paragraph's nearest neighbour comes from the
same Wikipedia article.

    uv run python src/embed.py --figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch

from data import DATASET_REVISION, load_squad

N_EXAMPLES = 5          # paragraphs whose neighbours we print
MAX_LEN = {"minilm": 256, "smollm2": 512}


def cache_dir() -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "mlcourse" / f"squad-{DATASET_REVISION[:7]}"


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of hidden (batch, seq, d) over the positions where mask (batch, seq) is 1."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


@torch.inference_mode()
def encode(model, tok, texts: list[str], max_len: int, batch: int, device, last_hidden) -> tuple[np.ndarray, int]:
    """Unit vectors, one per text, in input order; also how many texts were cut at max_len."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs: dict[int, np.ndarray] = {}
    cut = 0
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt", return_overflowing_tokens=False, return_length=False)
        full = tok([texts[i] for i in idx], truncation=False)["input_ids"]
        cut += sum(len(f) > max_len for f in full)
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        h = last_hidden(model, enc)
        v = torch.nn.functional.normalize(mean_pool(h.float(), enc["attention_mask"]), dim=-1)
        for i, row in zip(idx, v.cpu().numpy()):
            vecs[i] = row
    out = np.stack([vecs[i] for i in range(len(texts))]).astype(np.float32)
    return out, cut


def minilm_hidden(model, enc):
    return model(**enc).last_hidden_state


def smollm_hidden(model, enc):
    return model.model(**enc).last_hidden_state      # the decoder body, after its final norm


def embed_all(name: str, data: dict, g: common.Guard) -> dict:
    device = common.get_device()
    paras = [p["text"] for p in data["paragraphs"]]
    qs = [q["question"] for q in data["questions"]]
    if name == "minilm":
        model, tok = common.load_embedder()
        fn, bp, bq = minilm_hidden, 32, 128
    else:
        model, tok = common.load_hf_model(), common.load_tokenizer()
        tok.padding_side = "right"
        fn, bp, bq = smollm_hidden, 8, 64
    P, cut_p = encode(model, tok, paras, MAX_LEN[name], bp, device, fn)
    g.tick(f"{name}_paragraphs")
    Q, cut_q = encode(model, tok, qs, MAX_LEN[name], bq, device, fn)
    g.tick(f"{name}_questions")
    check = None
    if name == "minilm":
        # The same eight paragraphs on MPS and on the CPU.
        model_cpu = model.to("cpu")
        P_cpu, _ = encode(model_cpu, tok, paras[:8], MAX_LEN[name], 8, torch.device("cpu"), fn)
        check = {"texts": 8, "max_abs_diff": float(np.abs(P_cpu - P[:8]).max()),
                 "min_cosine": float((P_cpu * P[:8]).sum(1).min()), "device": str(device)}
    params = sum(p.numel() for p in model.parameters())
    del model
    common.free()
    return {"P": P, "Q": Q, "cut_paragraphs": cut_p, "cut_questions": cut_q, "params": params,
            "dim": P.shape[1], "mps_vs_cpu": check}


def describe(P: np.ndarray, Q: np.ndarray, data: dict, rng: np.random.Generator) -> dict:
    gold = np.array([q["gold"] for q in data["questions"]])
    titles = np.array([p["title"] for p in data["paragraphs"]])
    cos_gold = (Q * P[gold]).sum(1)
    rand = rng.integers(len(P), size=len(Q))
    cos_rand = (Q * P[rand]).sum(1)
    S = P @ P.T
    np.fill_diagonal(S, -np.inf)
    nn1 = S.argmax(1)
    same = float((titles[nn1] == titles).mean())
    # Pairs of distinct paragraphs, for the spread of cosines between unrelated texts.
    a, b = rng.integers(len(P), size=20000), rng.integers(len(P), size=20000)
    keep = a != b
    pp = (P[a[keep]] * P[b[keep]]).sum(1)
    q = lambda x: {k: round(float(np.quantile(x, v)), 4) for k, v in (("p05", .05), ("median", .5), ("p95", .95))}
    return {
        "cos_question_gold": {**q(cos_gold), "mean": round(float(cos_gold.mean()), 4)},
        "cos_question_random": {**q(cos_rand), "mean": round(float(cos_rand.mean()), 4)},
        "cos_paragraph_pairs": {**q(pp), "mean": round(float(pp.mean()), 4)},
        "gold_beats_random": round(float((cos_gold > cos_rand).mean()), 4),
        "nn_same_article": round(same, 4),
        "hist": {"gold": np.histogram(cos_gold, bins=60, range=(-0.2, 1.0))[0].tolist(),
                 "random": np.histogram(cos_rand, bins=60, range=(-0.2, 1.0))[0].tolist(),
                 "range": [-0.2, 1.0]},
        "_S": S,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    g = common.Guard("embed")
    data = load_squad()
    g.tick("load")
    titles = [p["title"] for p in data["paragraphs"]]
    counts = np.unique(titles, return_counts=True)[1]
    n = len(titles)
    chance_same = float(((counts * (counts - 1)).sum()) / (n * (n - 1)))

    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = {"data": data["info"], "chance_same_article": round(chance_same, 4), "models": {}}
    S_by = {}
    for name in ("minilm", "smollm2"):
        e = embed_all(name, data, g)
        np.save(d / f"{name}_paragraphs.npy", e["P"].astype(np.float16))
        np.save(d / f"{name}_questions.npy", e["Q"].astype(np.float16))
        desc = describe(e["P"], e["Q"], data, np.random.default_rng(0))
        S_by[name] = desc.pop("_S")
        out["models"][name] = {"id": common.EMBED_ID if name == "minilm" else common.MODEL_ID,
                               "revision": common.EMBED_REVISION if name == "minilm" else common.MODEL_REVISION,
                               "params": e["params"], "dim": e["dim"], "max_len": MAX_LEN[name],
                               "cut_paragraphs": e["cut_paragraphs"], "cut_questions": e["cut_questions"], **desc}
        if e["mps_vs_cpu"]:
            out["mps_vs_cpu"] = {"model": "minilm", **e["mps_vs_cpu"]}
        print(f"{name}: gold cos median {desc['cos_question_gold']['median']}, random "
              f"{desc['cos_question_random']['median']}, nn same article {desc['nn_same_article']}")

    # Neighbours of a few paragraphs, picked by a seeded shuffle, not by hand.
    rng = np.random.default_rng(0)
    picks = rng.permutation(n)[:N_EXAMPLES]
    ex = []
    for p in picks:
        row = {"paragraph": int(p), "title": titles[p], "text": data["paragraphs"][p]["text"][:300]}
        for name in ("minilm", "smollm2"):
            top = np.argsort(-S_by[name][p])[:3]
            row[name] = [{"paragraph": int(j), "title": titles[j], "cos": round(float(S_by[name][p, j]), 4),
                          "text": data["paragraphs"][j]["text"][:160]} for j in top]
        ex.append(row)
    out["neighbours"] = ex
    out["cache"] = f"~/.cache/huggingface/mlcourse/squad-{DATASET_REVISION[:7]}/"
    out["guard"] = g.report()
    common.write_results("embed", out)
    if args.figures:
        import figures
        figures.fig_embed()


if __name__ == "__main__":
    main()
