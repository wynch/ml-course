"""Section 3: retrieve, then generate. SmolLM2-360M-Instruct answers 200 SQuAD questions.

Four conditions, the same 200 questions (each on a different paragraph, see
data.rag_sample), greedy decoding, at most 32 new tokens:

- **none**: the question alone;
- **bm25**: the question and BM25's top-1 paragraph from all 2,067;
- **minilm**: the question and MiniLM's top-1 paragraph from all 2,067;
- **gold**: the question and the paragraph it was written on, the ceiling
  for any retriever with this reader and this prompt.

Generation and scoring are separate steps. The first writes every reply to
results/rag_generations.json; the second reads only that file, so the scores
can be checked or recomputed with no model. A second run reuses the file
unless --regenerate is given.

    uv run python src/rag.py --figures
"""

from __future__ import annotations

import argparse
import json

import common
import numpy as np
import torch

import squad_metric
from bm25 import BM25
from data import load_squad, rag_sample
from embed import cache_dir

CONDITIONS = ("none", "bm25", "minilm", "gold")
MAX_NEW = 32
BATCH = 16
GEN_FILE = common.RESULTS / "rag_generations.json"

PROMPT_CTX = ("Read the passage, then answer the question with a short phrase copied from the passage.\n\n"
              "Passage: {passage}\n\nQuestion: {question}")
PROMPT_NONE = "Answer the question with a short phrase.\n\nQuestion: {question}"
# One worked example about a made-up town, written for this module (not from SQuAD), shown as an
# earlier chat turn so the reply comes as a short phrase rather than a sentence. Chosen on six
# questions outside the 200-question sample.
SHOT_PASSAGE = ("The made-up town of Arlen built its first bridge of wood in 1388. A stone bridge replaced it in "
                "1861, and the current steel bridge opened in 1956 after a flood had destroyed the old one.")
SHOT_QUESTION = "In what year did the current bridge open?"
SHOT_ANSWER = "1956"


def extract(reply: str) -> str:
    """The first non-empty line, without a leading 'Answer:' and a trailing full stop."""
    line = next((l.strip() for l in reply.strip().splitlines() if l.strip()), "")
    if line.lower().startswith("answer:"):
        line = line[7:].strip()
    return line[:-1] if line.endswith(".") else line


def passages_for(data: dict, sample: list[int]) -> dict[str, list[int | None]]:
    paras = [p["text"] for p in data["paragraphs"]]
    qs = [data["questions"][i]["question"] for i in sample]
    index = BM25(paras)
    P = np.load(cache_dir() / "minilm_paragraphs.npy").astype(np.float32)
    Q = np.load(cache_dir() / "minilm_questions.npy").astype(np.float32)[sample]
    return {"none": [None] * len(sample),
            "bm25": [int(np.argmax(index.scores(q))) for q in qs],
            "minilm": [int(j) for j in (Q @ P.T).argmax(1)],
            "gold": [data["questions"][i]["gold"] for i in sample]}


def user_text(question: str, passage: str | None) -> str:
    if passage is None:
        return PROMPT_NONE.format(question=question)
    return PROMPT_CTX.format(passage=passage, question=question)


def prompt_ids(tok, question: str, passage: str | None) -> str:
    shot = user_text(SHOT_QUESTION, None if passage is None else SHOT_PASSAGE)
    msgs = [{"role": "user", "content": shot}, {"role": "assistant", "content": SHOT_ANSWER},
            {"role": "user", "content": user_text(question, passage)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate(model, tok, prompts: list[str]) -> list[str]:
    device = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: dict[int, str] = {}
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            out[i] = tok.decode(row, skip_special_tokens=True)
    return [out[i] for i in range(len(prompts))]


@torch.inference_mode()
def mps_vs_cpu(model, tok, prompt: str) -> dict:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    a = model(**enc.to("mps")).logits[0].float().cpu()
    model.to("cpu")
    b = model(**enc.to("cpu")).logits[0].float()
    model.to("mps")
    return {"tokens": int(enc["input_ids"].shape[1]), "max_abs_diff_logits": float((a - b).abs().max()),
            "logit_range": [float(b.min()), float(b.max())],
            "same_top1_every_position": bool((a.argmax(-1) == b.argmax(-1)).all())}


def run_generation(data: dict, sample: list[int], g: common.Guard) -> dict:
    chosen = passages_for(data, sample)
    g.tick("retrieve")
    tok = common.load_tokenizer()
    tok.padding_side = "left"
    model = common.load_hf_model()
    g.tick("load")
    gens = {"model": common.MODEL_ID, "revision": common.MODEL_REVISION, "max_new_tokens": MAX_NEW,
            "decoding": "greedy", "prompt_context": PROMPT_CTX, "prompt_none": PROMPT_NONE,
            "one_shot": {"passage": SHOT_PASSAGE, "question": SHOT_QUESTION, "answer": SHOT_ANSWER},
            "question_ids": [data["questions"][i]["id"] for i in sample], "conditions": {}}
    for c in CONDITIONS:
        prompts = [prompt_ids(tok, data["questions"][i]["question"],
                              None if p is None else data["paragraphs"][p]["text"])
                   for i, p in zip(sample, chosen[c])]
        replies = generate(model, tok, prompts)
        gens["conditions"][c] = {"paragraph": chosen[c], "reply": replies}
        g.tick(f"generate_{c}")
        print(f"  generated {c}: {replies[0]!r}")
    gens["mps_vs_cpu"] = mps_vs_cpu(model, tok, prompt_ids(tok, data["questions"][sample[0]]["question"],
                                                           data["paragraphs"][chosen["gold"][0]]["text"]))
    g.tick("mps_vs_cpu")
    del model
    common.free()
    GEN_FILE.write_text(json.dumps(gens, indent=1, ensure_ascii=False) + "\n")
    print(f"  wrote {GEN_FILE.relative_to(common.MODULE)}")
    return gens


def score(data: dict, sample: list[int], gens: dict) -> dict:
    """Exact match and F1 per condition, from the cached replies only."""
    by_id = {q["id"]: q for q in data["questions"]}
    qs = [by_id[i] for i in gens["question_ids"]]
    assert [q["id"] for q in qs] == [data["questions"][i]["id"] for i in sample]
    gold = np.array([q["gold"] for q in qs])
    rows, per = {}, {}
    for c, v in gens["conditions"].items():
        preds = [extract(r) for r in v["reply"]]
        em, f1 = map(np.array, zip(*(squad_metric.best(p, q["answers"]) for p, q in zip(preds, qs))))
        row = {"em": round(float(em.mean()), 4), "f1": round(float(f1.mean()), 4),
               "em_se": round(float(em.std(ddof=1) / np.sqrt(len(em))), 4),
               "mean_reply_words": round(float(np.mean([len(p.split()) for p in preds])), 2)}
        if c in ("bm25", "minilm"):
            hit = np.array(v["paragraph"]) == gold
            row["top1_is_gold"] = int(hit.sum())
            row["em_when_gold"] = round(float(em[hit].mean()), 4)
            row["em_when_not_gold"] = round(float(em[~hit].mean()), 4) if (~hit).any() else None
            row["f1_when_gold"] = round(float(f1[hit].mean()), 4)
            row["f1_when_not_gold"] = round(float(f1[~hit].mean()), 4) if (~hit).any() else None
        rows[c] = row
        per[c] = {"pred": preds, "em": em.tolist(), "f1": [round(x, 4) for x in f1.tolist()]}
    # Answer appears verbatim in the passage shown: a crude check of what the reader could copy.
    norm = squad_metric.normalize
    for c in ("bm25", "minilm", "gold"):
        shown = [data["paragraphs"][p]["text"] for p in gens["conditions"][c]["paragraph"]]
        rows[c]["answer_in_passage"] = int(sum(any(norm(a) in norm(t) for a in q["answers"]) for q, t in zip(qs, shown)))
    return {"table": rows, "per_question": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()
    g = common.Guard("rag")
    data = load_squad()
    sample = rag_sample(data)
    if args.regenerate or not GEN_FILE.exists():
        gens = run_generation(data, sample, g)
        generated = True
    else:
        gens = json.loads(GEN_FILE.read_text())
        generated = False
        print(f"  read cached replies from {GEN_FILE.name}; --regenerate to run the model")
    s = score(data, sample, gens)
    for c, r in s["table"].items():
        print(f"{c:7s} EM {r['em']:.3f}  F1 {r['f1']:.3f}")
    ex = []
    rng = np.random.default_rng(2)
    for k in rng.permutation(len(sample))[:6]:
        q = data["questions"][sample[k]]
        ex.append({"question": q["question"], "answers": q["answers"],
                   **{c: {"reply": gens["conditions"][c]["reply"][k], "pred": s["per_question"][c]["pred"][k],
                          "em": s["per_question"][c]["em"][k], "f1": s["per_question"][c]["f1"][k],
                          "paragraph": gens["conditions"][c]["paragraph"][k]} for c in CONDITIONS}})
    out = {"n": len(sample), "conditions": list(CONDITIONS), "table": s["table"],
           "per_question": s["per_question"], "examples": ex, "mps_vs_cpu": gens["mps_vs_cpu"],
           "generation_file": GEN_FILE.name}
    if generated:
        out["guard"] = g.report()
    else:
        out["guard"] = common.read_results("rag")["guard"]      # the generating run's cost
    common.write_results("rag", out)
    if args.figures:
        import figures
        figures.fig_rag()


if __name__ == "__main__":
    main()
