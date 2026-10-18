"""Generate every output the module reads, and cache it to results/generations/.

    uv run python src/generate.py STAGE [--budget SECONDS]

    direct   8-shot answer-only prompt, greedy, 16 tokens       -> direct.jsonl
    greedy   8-shot chain-of-thought prompt, greedy, 256 tokens -> greedy.jsonl
    sample   same prompt, 8 samples at temperature 0.7          -> samples.jsonl
    verify   the model grades each sample, P(Yes) vs P(No)      -> verify.jsonl

The committed caches took about 45 minutes in all on an M5. A stage picks up
where its file stops and exits once --budget seconds (default 270) have
passed, so you can run it in chunks; run it again until it prints
"remaining": 0. To start a stage over, delete its file first. analyse.py
reads only these files; it never loads the model.

Qwen3 can "think" (write a hidden reasoning block) before it answers. We turn
that off: the chat template is rendered with enable_thinking=False, which ends
the prompt with an empty <think></think> block.
"""

from __future__ import annotations

import argparse
import json
import math
import time

from common import GEN, MODEL_ID, MODEL_REVISION, Guard, read_jsonl
import gsm8k

N_SAMPLES = 8
TEMP = 0.7
MAX_COT = 256
MAX_DIRECT = 16
SEED_BASE = 15000
# The model ignores answer-only examples and starts to reason anyway, so the
# answer-only condition starts the reply for it.
DIRECT_FORCE = "The answer is"

VERIFY_TEMPLATE = (
    "Here is a math word problem and a proposed solution.\n\n"
    "Problem: {question}\n\n"
    "Proposed solution:\n{solution}\n\n"
    "Is the proposed solution correct, including its final answer? "
    "Reply with only Yes or No."
)


def problems() -> list[dict]:
    path = GEN / "problems.jsonl"
    if not path.exists():
        GEN.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i, r in enumerate(gsm8k.load_split("test")[: gsm8k.N_PROBLEMS]):
                f.write(json.dumps({"id": i, "question": r["question"], "solution": r["answer"],
                                    "gold": gsm8k.gold(r["answer"])}, ensure_ascii=False) + "\n")
    return read_jsonl(path)


class Runner:
    def __init__(self, budget: float):
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        self.mx, self.budget, self.t0 = mx, budget, time.time()
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt"])
        self.model, self.tok = load(path)
        self.enc = self.tok._tokenizer          # the Hugging Face tokenizer inside mlx-lm's wrapper
        self.shots = gsm8k.shots()
        self.problems = problems()

    def render(self, msgs: list[dict]) -> str:
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def split_prompt(self, kind: str, question: str, force: str = "") -> tuple[list[int], list[int]]:
        """Token ids of the 8 examples (the same for every problem) and of the rest."""
        full = self.render(gsm8k.messages(self.shots, kind, question)) + force
        prefix = full[: full.rindex("<|im_start|>user")]
        p_ids = self.enc.encode(prefix, add_special_tokens=False)
        s_ids = self.enc.encode(full[len(prefix):], add_special_tokens=False)
        assert p_ids + s_ids == self.enc.encode(full, add_special_tokens=False)
        return p_ids, s_ids

    def prefix_cache(self, p_ids: list[int]):
        """Run the 8 examples once; every problem reuses their keys and values."""
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        self.model(self.mx.array([p_ids]), cache=cache)
        self.mx.eval([c.state for c in cache])
        return cache

    def out_of_time(self) -> bool:
        return time.time() - self.t0 > self.budget

    def greedy(self, kind: str, max_tokens: int, out, force: str = "") -> int:
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import trim_prompt_cache

        have = {r["id"] for r in read_jsonl(out)}
        todo = [p for p in self.problems if p["id"] not in have]
        cache = self.prefix_cache(self.split_prompt(kind, self.problems[0]["question"])[0])
        base = cache[0].offset
        eos = set(self.tok.eos_token_ids)
        done = 0
        with open(out, "a") as f:
            for p in todo:
                if self.out_of_time():
                    break
                _, s_ids = self.split_prompt(kind, p["question"], force)
                toks, finish = [], "length"
                for (t, _), _n in zip(generate_step(self.mx.array(s_ids), self.model,
                                                     prompt_cache=cache, max_tokens=max_tokens),
                                      range(max_tokens)):
                    if int(t) in eos:
                        finish = "stop"
                        break
                    toks.append(int(t))
                trim_prompt_cache(cache, cache[0].offset - base)   # back to the 8 examples
                f.write(json.dumps({"id": p["id"], "text": force + self.tok.decode(toks),
                                    "tokens": len(toks), "finish": finish}, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
        return len(todo) - done

    def sample(self, out) -> int:
        from mlx_lm.generate import batch_generate
        from mlx_lm.sample_utils import make_sampler

        have = {r["id"] for r in read_jsonl(out)}
        todo = [p for p in self.problems if p["id"] not in have]
        cache = self.prefix_cache(self.split_prompt("cot", self.problems[0]["question"])[0])
        sampler = make_sampler(temp=TEMP)
        done = 0
        with open(out, "a") as f:
            for p in todo:
                if self.out_of_time():
                    break
                _, s_ids = self.split_prompt("cot", p["question"])
                self.mx.random.seed(SEED_BASE + p["id"])
                res = batch_generate(self.model, self.tok, [s_ids] * N_SAMPLES,
                                     prompt_caches=[cache] * N_SAMPLES, max_tokens=MAX_COT,
                                     sampler=sampler, return_token_ids=True)
                samples = [{"text": self.tok.decode([int(t) for t in ids]), "tokens": len(ids),
                            "finish": "length" if len(ids) >= MAX_COT else "stop"}
                           for ids in res.token_ids]
                f.write(json.dumps({"id": p["id"], "samples": samples}, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
        return len(todo) - done

    def verify(self, out) -> int:
        mx = self.mx
        yes = self.enc.encode("Yes", add_special_tokens=False)
        no = self.enc.encode("No", add_special_tokens=False)
        assert len(yes) == 1 and len(no) == 1
        samples = {r["id"]: r["samples"] for r in read_jsonl(GEN / "samples.jsonl")}
        have = {r["id"] for r in read_jsonl(out)}
        todo = [p for p in self.problems if p["id"] in samples and p["id"] not in have]
        done = 0
        with open(out, "a") as f:
            for p in todo:
                if self.out_of_time():
                    break
                rows = []
                for s in samples[p["id"]]:
                    msg = [{"role": "user", "content": VERIFY_TEMPLATE.format(
                        question=p["question"], solution=s["text"].strip())}]
                    ids = self.enc.encode(self.render(msg), add_special_tokens=False)
                    logits = self.model(mx.array([ids]))[0, -1].astype(mx.float32)
                    lp = logits - mx.logsumexp(logits)
                    ly, ln = lp[yes[0]].item(), lp[no[0]].item()
                    rows.append({"lp_yes": round(ly, 5), "lp_no": round(ln, 5),
                                 "score": round(1 / (1 + math.exp(ln - ly)), 6),
                                 "top": self.tok.decode([int(mx.argmax(lp).item())]),
                                 "prompt_tokens": len(ids)})
                f.write(json.dumps({"id": p["id"], "verdicts": rows}, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
        return len(todo) - done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["direct", "greedy", "sample", "verify"])
    ap.add_argument("--budget", type=float, default=270.0)
    a = ap.parse_args()
    guard = Guard(f"generate {a.stage}")
    r = Runner(a.budget)
    if a.stage == "direct":
        left = r.greedy("direct", MAX_DIRECT, GEN / "direct.jsonl", force=DIRECT_FORCE)
    elif a.stage == "greedy":
        left = r.greedy("cot", MAX_COT, GEN / "greedy.jsonl")
    elif a.stage == "sample":
        left = r.sample(GEN / "samples.jsonl")
    else:
        left = r.verify(GEN / "verify.jsonl")
    print(json.dumps({"stage": a.stage, "remaining": left, **guard.report()}))


if __name__ == "__main__":
    main()
