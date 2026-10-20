"""Section 4: the library way. TRL's GRPOTrainer with a LoRA adapter on
SmolLM2-360M-Instruct, on the same 3-digit sums and the same checker rule.

Stages, each under five minutes, each picking up where the last stopped:

    uv run python src/library.py base      # the start model: greedy, and 16 samples per problem
    uv run python src/library.py train     # repeat until it prints "remaining steps: 0"
    uv run python src/library.py after     # the trained adapter, measured the same way

The model is the one module 17 uses, at its pinned revision, so nothing new
is downloaded if module 17 ran. It runs in bfloat16 on the GPU (MPS) under the
module's watermark; checkpoints go to ``outputs/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import re
import time

import common  # noqa: F401  (laptop guard before torch)
import numpy as np
import torch

import task
from common import MODULE, RESULTS, Guard, read_results, write_results

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"     # module 17's pin
OUT = MODULE / "python" / "outputs" / "grpo_lora"
N_EVAL_PROBLEMS = 200        # the first 200 of the 500 test problems
N_SAMPLES = 16
MAX_NEW = 16
# Prompts per generate call; 16 samples each. At 25 prompts the base stage ran
# out of MPS memory at the 5.92 GB cap.
BATCH = 5
STEPS = 120
CHUNK_SECONDS = 270
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# bfloat16 base weights; PEFT keeps the LoRA weights in float32. In float32 with
# 32 replies per backward pass, a first try peaked at 2.47 GB RSS plus 3.62 GB
# on the GPU, over the module's 4 GB budget.
DTYPE = torch.bfloat16


def question(a: int, b: int) -> str:
    return f"What is {a}+{b}? Reply with the number only."


def last_number(text: str) -> str | None:
    found = re.findall(r"\d[\d,]*", text)
    return found[-1].rstrip(",").replace(",", "") if found else None


def check(text: str, a: int, b: int) -> int:
    """The checker: 1 if the last number in the reply is a + b. Commas inside the number are ignored.

    A first try read the first number, and scored the base 0% greedy: it repeats
    the question ("865 + 673 = 1548") before the sum. Kept in
    results/library_base_first_number.json.
    """
    return int(last_number(text) == str(a + b))


def reward_fn(completions, a, b, **_):
    """TRL calls this with the batch's completions and the dataset's extra columns."""
    out = []
    for c, x, y in zip(completions, a, b):
        text = c[0]["content"] if isinstance(c, list) else c
        out.append(float(check(text, int(x), int(y))))
    return out


def load(adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=DTYPE)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.to(DEVICE).eval(), tok


@torch.no_grad()
def measure(model, tok, g: Guard) -> dict:
    test, _ = task.splits()
    test = test[:N_EVAL_PROBLEMS]
    chats = [tok.apply_chat_template([{"role": "user", "content": question(int(a), int(b))}],
                                     tokenize=False, add_generation_prompt=True) for a, b in test]
    greedy, right, lengths, examples = [], [], [], []
    torch.manual_seed(0)
    for i in range(0, len(chats), BATCH):
        enc = tok(chats[i:i + BATCH], return_tensors="pt", padding=True).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        greedy += [check(t, int(a), int(b)) for t, (a, b) in zip(texts, test[i:i + BATCH])]
        examples += texts[:1]
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_k=0,
                             num_return_sequences=N_SAMPLES, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        lengths += (gen != tok.pad_token_id).sum(1).tolist()
        for j, (a, b) in enumerate(test[i:i + BATCH]):
            right.append(sum(check(t, int(a), int(b)) for t in texts[j * N_SAMPLES:(j + 1) * N_SAMPLES]))
        g.tick()
        if DEVICE == "mps":
            torch.mps.empty_cache()
    ks = [1, 2, 4, 8, 16]
    return {
        "problems": len(test), "samples": N_SAMPLES,
        "greedy_acc": float(np.mean(greedy)),
        "right_per_problem": right,
        "pass_at_k": {str(k): float(np.mean([task.pass_at_k(N_SAMPLES, c, k) for c in right])) for k in ks},
        "mean_reply_tokens": float(np.mean(lengths)),
        "greedy_examples": examples[:8],
    }


def train(g: Guard) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    _, draw = task.splits()
    pairs = draw(STEPS * 4, np.random.default_rng(7))
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": question(int(a), int(b))}] for a, b in pairs],
        "a": [int(a) for a, _ in pairs], "b": [int(b) for _, b in pairs]})

    cfg = GRPOConfig(
        output_dir=str(OUT), max_steps=STEPS, learning_rate=1e-4, lr_scheduler_type="constant",
        per_device_train_batch_size=16, gradient_accumulation_steps=2,   # 32 replies per step, backward in halves
        num_generations=8,                                         # 4 problems x 8 replies per step
        max_completion_length=MAX_NEW, temperature=1.0, top_k=0,
        loss_type="grpo", scale_rewards="group", beta=0.0,         # section 2's grpo arm
        logging_steps=5, save_steps=10, save_total_limit=2, report_to=[], seed=0,
        use_cpu=DEVICE == "cpu", bf16=False, fp16=False, shuffle_dataset=False,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    t0 = time.perf_counter()

    class Chunk(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            g.tick()
            if time.perf_counter() - t0 > CHUNK_SECONDS - 30:
                control.should_save = True
                control.should_training_stop = True

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=DTYPE)
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=lora, callbacks=[Chunk()])
    ckpts = sorted(OUT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    step = trainer.state.global_step
    if step >= STEPS:
        trainer.save_model(str(OUT / "final"))
    return {"global_step": step, "log_history": trainer.state.log_history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "train", "after"])
    args = ap.parse_args()
    g = Guard(f"library_{args.stage}", device=DEVICE)
    if args.stage == "base":
        model, tok = load()
        write_results("library_base", {"model": MODEL_ID, "revision": MODEL_REVISION, "checker": "last number",
                                       "measure": measure(model, tok, g), "guard": g.report()})
    elif args.stage == "after":
        model, tok = load(str(OUT / "final"))
        write_results("library_after", {"model": MODEL_ID, "revision": MODEL_REVISION, "adapter": "LoRA r=16",
                                        "checker": "last number",
                                        "measure": measure(model, tok, g), "guard": g.report()})
    else:
        r = train(g)
        prev = read_results("library_train") if (RESULTS / "library_train.json").exists() else {"chunks": []}
        prev["chunks"].append({"reached_step": r["global_step"], "guard": g.report()})
        prev["log_history"] = r["log_history"]
        prev["config"] = {"steps": STEPS, "problems_per_step": 4, "group": 8, "lr": 1e-4, "lora_r": 16,
                          "lora_alpha": 32, "loss_type": "grpo", "scale_rewards": "group", "beta": 0.0,
                          "max_completion_length": MAX_NEW, "temperature": 1.0,
                          "dtype": "bfloat16", "per_device_batch": 16, "grad_accum": 2}
        write_results("library_train", prev)
        print(f"remaining steps: {max(0, STEPS - r['global_step'])}")


if __name__ == "__main__":
    main()
