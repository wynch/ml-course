"""Section 3: DPO with TRL + LoRA on SmolLM2-360M-Instruct, beta 0.1.

Two configs share every setting but the budget:

    uv run python src/dpo_train.py --config preview --figures   # under 5 minutes
    uv run python src/dpo_train.py --config full --figures      # 200 steps, about 15 minutes

Before training, on held-out pairs: the reference model's log-probabilities
of each chosen and rejected reply. After training: the policy's. The implicit
reward of a reply is beta * (log pi - log pi_ref); held-out accuracy is the
share of pairs where the chosen reply gets the larger one.

On held-out prompts it also generates a greedy reply before and after, and
measures each reply's length, its perplexity under the reference model, and
the score that section 1's reward head gives it.

Takes the GPU lock for the whole run. The adapter lands in python/outputs/,
which git ignores.
"""

from __future__ import annotations

import argparse
import math

from common import FIGURES, LAYER, OUTPUTS, Guard, free, get_device, load_hf_model, load_tokenizer, \
    read_results, write_results

import numpy as np
import torch
import torch.nn.functional as F

import data
from reward_head import chat_ids, residuals

BETA = 0.1
LR = 5e-5
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
CONFIGS = {
    "preview": {"steps": 40, "batch": 1, "accum": 4, "n_eval": 100, "n_gen": 16, "max_new_tokens": 160},
    "full": {"steps": 200, "batch": 1, "accum": 4, "n_eval": 200, "n_gen": 32, "max_new_tokens": 160},
}
EVAL_BATCH = 4
TRAIN_MAX_TOKENS = 384      # training pairs only: longer ones run out of the 5.9 GB MPS cap and time
GEN_BATCH = 16


@torch.no_grad()
def reply_logps(model, tok, texts: list[tuple[str, str]], device, batch: int = EVAL_BATCH) -> tuple[np.ndarray, np.ndarray]:
    """For each (prompt, reply): the sum of the reply's token log-probs, and its token count.

    The reply's tokens run from the end of the prompt's chat template to the end
    of the full template, so they include <|im_end|> and the newline after it,
    as in TRL.
    """
    enc = [chat_ids(tok, p, r) for p, r in texts]
    order = np.argsort([len(e[0]) for e in enc])
    sums, counts = np.zeros(len(texts)), np.zeros(len(texts), dtype=int)
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        L = max(len(enc[i][0]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for k, i in enumerate(idx):
            ids[k, :len(enc[i][0])] = torch.tensor(enc[i][0])
            mask[k, :len(enc[i][0])] = 1
        ids, mask = ids.to(device), mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        lp = torch.gather(F.log_softmax(logits, -1), -1, ids[:, 1:, None])[..., 0].cpu()
        for k, i in enumerate(idx):
            n, s = len(enc[i][0]), enc[i][1]
            sums[i] = float(lp[k, s - 1:n - 1].sum())
            counts[i] = n - s
        del logits
    return sums, counts


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new: int, device) -> list[dict]:
    tok.padding_side = "left"
    out = []
    for b0 in range(0, len(prompts), GEN_BATCH):
        chunk = prompts[b0:b0 + GEN_BATCH]
        enc = tok.apply_chat_template([[{"role": "user", "content": p}] for p in chunk], add_generation_prompt=True,
                                      padding=True, return_tensors="pt", return_dict=True).to(device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for k in range(len(chunk)):
            new = g[k, enc["input_ids"].shape[1]:].tolist()
            ended = tok.convert_tokens_to_ids("<|im_end|>") in new
            if ended:
                new = new[:new.index(tok.convert_tokens_to_ids("<|im_end|>"))]
            new = [t for t in new if t != tok.pad_token_id]
            out.append({"reply": tok.decode(new, skip_special_tokens=True).strip(), "tokens": len(new),
                        "finished": ended})
    tok.padding_side = "right"
    return out


def head_scores(base, tok, texts, device) -> np.ndarray:
    h = read_results("reward_head")["head"]
    last, _ = residuals(base, tok, texts, device, layer=LAYER)
    return ((last - np.array(h["mu"])) / np.array(h["sd"])) @ np.array(h["w"])


def acc(margin: np.ndarray) -> dict:
    return {"strict": round(float((margin > 0).mean()), 4), "ties": int((margin == 0).sum()),
            "ties_half": round(float(((margin > 0) + 0.5 * (margin == 0)).mean()), 4)}


def main() -> None:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import DPOConfig, DPOTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, default="preview")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    C = CONFIGS[args.config]
    guard = Guard(f"dpo_train_{args.config}")
    set_seed(0)
    tok = load_tokenizer()
    sp = data.splits(data.load_pairs(tok)[0])
    ev = sp["heldout"][:C["n_eval"]]
    gen_prompts = [p["prompt"] for p in sp["heldout"][-C["n_gen"]:]]   # disjoint from the eval pairs
    device = get_device()
    model = load_hf_model()
    guard.tick("load")

    # Before: the reference model. Its implicit reward is zero for every reply.
    ref_c, n_c = reply_logps(model, tok, [(p["prompt"], p["chosen"]) for p in ev], device)
    ref_r, n_r = reply_logps(model, tok, [(p["prompt"], p["rejected"]) for p in ev], device)
    guard.tick("eval_before")
    gen_before = generate(model, tok, gen_prompts, C["max_new_tokens"], device)
    guard.tick("gen_before")

    train = [p for p in sp["dpo_train"] if max(p["len_chosen"], p["len_rejected"]) <= TRAIN_MAX_TOKENS]
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": p["prompt"]}],
                             "chosen": [{"role": "assistant", "content": p["chosen"]}],
                             "rejected": [{"role": "assistant", "content": p["rejected"]}]} for p in train])
    cfg = DPOConfig(
        output_dir=str(OUTPUTS / f"dpo-{args.config}"), beta=BETA, learning_rate=LR,
        max_steps=C["steps"], per_device_train_batch_size=C["batch"], gradient_accumulation_steps=C["accum"],
        lr_scheduler_type="cosine", warmup_steps=10, logging_steps=5, max_length=TRAIN_MAX_TOKENS + 8,
        bf16=False, fp16=False, report_to=[], save_strategy="no", seed=0, dataset_num_proc=None,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=LoraConfig(task_type="CAUSAL_LM", **LORA))
    n_train, n_total = trainer.model.get_nb_trainable_parameters()
    guard.tick("setup")
    trainer.train()
    guard.tick("train")
    pm = trainer.model
    pm.eval()
    log = [{k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()}
           for h in trainer.state.log_history if "loss" in h]

    pol_c, _ = reply_logps(pm, tok, [(p["prompt"], p["chosen"]) for p in ev], device)
    pol_r, _ = reply_logps(pm, tok, [(p["prompt"], p["rejected"]) for p in ev], device)
    guard.tick("eval_after")
    gen_after = generate(pm, tok, gen_prompts, C["max_new_tokens"], device)
    guard.tick("gen_after")

    with pm.disable_adapter():
        base = pm.get_base_model()
        ppl = {}
        for name, gens in (("before", gen_before), ("after", gen_after)):
            lp, n = reply_logps(base, tok, [(q, g["reply"]) for q, g in zip(gen_prompts, gens)], device)
            ppl[name] = np.exp(-lp / n)
        hs = {name: head_scores(base, tok, [(q, g["reply"]) for q, g in zip(gen_prompts, gens)], device)
              for name, gens in (("before", gen_before), ("after", gen_after))}
        # The adapter must really be off: the reference log-probs again on the first pairs.
        ref_again, _ = reply_logps(base, tok, [(p["prompt"], p["chosen"]) for p in ev[:8]], device)
    guard.tick("score")
    pm.save_pretrained(str(OUTPUTS / f"dpo-{args.config}" / "adapter"))
    del trainer, pm, model, base
    free()

    r_c, r_r = BETA * (pol_c - ref_c), BETA * (pol_r - ref_r)
    margin_after = r_c - r_r
    win = hs["after"] - hs["before"]
    res = {
        "config": args.config, "settings": dict(C, beta=BETA, learning_rate=LR, lora=LORA, schedule="cosine",
                                                 warmup_steps=10, effective_batch=C["batch"] * C["accum"],
                                                 train_pairs_available=len(train), train_max_tokens=TRAIN_MAX_TOKENS,
                                                 pairs_seen=C["steps"] * C["batch"] * C["accum"]),
        "trainable_params": int(n_train), "total_params": int(n_total),
        "heldout": {
            "pairs": len(ev),
            "implicit_reward_accuracy_before": acc(np.zeros(len(ev))),
            "implicit_reward_accuracy_after": acc(margin_after),
            "mean_margin_after": round(float(margin_after.mean()), 4),
            "mean_reward_chosen_after": round(float(r_c.mean()), 4),
            "mean_reward_rejected_after": round(float(r_r.mean()), 4),
            "reference_logp_accuracy": acc(ref_c - ref_r),
            "reference_logp_per_token_accuracy": acc(ref_c / n_c - ref_r / n_r),
            "logp_change_chosen": round(float((pol_c - ref_c).mean()), 3),
            "logp_change_rejected": round(float((pol_r - ref_r).mean()), 3),
            "longer_reply_wins": acc((n_c - n_r).astype(float)),
            "after_agrees_with_longer": round(float(((margin_after > 0) == (n_c > n_r))[n_c != n_r].mean()), 4),
            "adapter_off_check_max_abs_diff": float(np.abs(ref_again - ref_c[:8]).max()),
            "per_pair": [{"id": p["id"], "margin": round(float(m), 4), "ref_logp_diff": round(float(d), 3),
                          "tokens_chosen": int(a), "tokens_rejected": int(b)}
                         for p, m, d, a, b in zip(ev, margin_after, ref_c - ref_r, n_c, n_r)],
        },
        "generations": {
            "prompts": len(gen_prompts), "max_new_tokens": C["max_new_tokens"],
            "before": {"tokens_mean": round(float(np.mean([g["tokens"] for g in gen_before])), 1),
                       "tokens_median": float(np.median([g["tokens"] for g in gen_before])),
                       "finished": sum(g["finished"] for g in gen_before),
                       "ppl_ref_median": round(float(np.median(ppl["before"])), 3),
                       "ppl_ref_mean": round(float(np.mean(ppl["before"])), 3),
                       "head_score_mean": round(float(hs["before"].mean()), 3)},
            "after": {"tokens_mean": round(float(np.mean([g["tokens"] for g in gen_after])), 1),
                      "tokens_median": float(np.median([g["tokens"] for g in gen_after])),
                      "finished": sum(g["finished"] for g in gen_after),
                      "ppl_ref_median": round(float(np.median(ppl["after"])), 3),
                      "ppl_ref_mean": round(float(np.mean(ppl["after"])), 3),
                      "head_score_mean": round(float(hs["after"].mean()), 3)},
            "head_win_rate_after_vs_before": round(float(((win > 0) + 0.5 * (win == 0)).mean()), 4),
            "identical_replies": sum(a["reply"] == b["reply"] for a, b in zip(gen_before, gen_after)),
            "samples": [{"prompt": q, "before": b["reply"], "after": a["reply"],
                         "tokens": [b["tokens"], a["tokens"]],
                         "ppl_ref": [round(float(x), 3), round(float(y), 3)],
                         "head": [round(float(u), 3), round(float(v), 3)]}
                        for q, b, a, x, y, u, v in zip(gen_prompts, gen_before, gen_after, ppl["before"],
                                                       ppl["after"], hs["before"], hs["after"])],
        },
        "log_history": log,
        "guard": guard.report(),
    }
    h = res["heldout"]
    print(f"held-out implicit-reward accuracy: before {h['implicit_reward_accuracy_before']['strict']} "
          f"(all ties), after {h['implicit_reward_accuracy_after']['strict']}; reference log-prob accuracy "
          f"{h['reference_logp_accuracy']['strict']}")
    g = res["generations"]
    print(f"replies: tokens {g['before']['tokens_mean']} -> {g['after']['tokens_mean']}, ppl under reference "
          f"{g['before']['ppl_ref_median']} -> {g['after']['ppl_ref_median']}, head win rate {g['head_win_rate_after_vs_before']}")
    write_results(f"dpo_{args.config}", res)
    if args.figures:
        fig_dpo(res, FIGURES / f"dpo_{args.config}.png")


def fig_dpo(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = r["log_history"]
    st = [x["step"] for x in log]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.6))
    a.plot(st, [x["loss"] for x in log], color="#b7410e", marker="o", ms=3)
    a.axhline(math.log(2), color="#999999", lw=0.8, ls="--")
    a.text(st[0], math.log(2) + 0.005, "log 2", fontsize=8, color="#777777")
    a.set_title("DPO loss (training batches)", fontsize=10)
    a.set_xlabel("step")
    b.plot(st, [x["rewards/chosen"] for x in log], color="#b7410e", label="chosen")
    b.plot(st, [x["rewards/rejected"] for x in log], color="#555555", label="rejected")
    b.axhline(0, color="#999999", lw=0.8)
    b.set_title("Implicit reward β·log(π/π_ref), training batches", fontsize=10)
    b.set_xlabel("step")
    b.legend(fontsize=8, frameon=False)
    h, g = r["heldout"], r["generations"]
    names = ["ref log-prob\n(before)", "implicit reward\n(after)", "longer\nwins"]
    vals = [h["reference_logp_accuracy"]["strict"], h["implicit_reward_accuracy_after"]["strict"],
            h["longer_reply_wins"]["ties_half"]]
    c.bar(names, vals, color=["#888888", "#b7410e", "#cccccc"])
    for k, v in enumerate(vals):
        c.text(k, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    c.axhline(0.5, color="#999999", lw=0.8, ls="--")
    c.set_ylim(0, 1)
    c.set_title(f"Held-out pair accuracy ({h['pairs']} pairs)", fontsize=10)
    for ax in (a, b, c):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{r['config']}: {r['settings']['steps']} steps, β {r['settings']['beta']}; replies "
                 f"{g['before']['tokens_mean']:.0f} → {g['after']['tokens_mean']:.0f} tokens, "
                 f"perplexity under the reference {g['before']['ppl_ref_median']:.2f} → {g['after']['ppl_ref_median']:.2f}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
