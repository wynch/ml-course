"""Section 2: the DPO loss by hand, checked against TRL on one batch.

DPO's loss for one pair, with policy pi and frozen reference pi_ref:

    loss = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                                - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

where log pi(y|x) is the sum of the log-probabilities of the reply's tokens.
This script computes it from logits with a few lines of torch, and compares
it with what TRL's DPOTrainer computes on the same eight held-out pairs.

The policy is SmolLM2-360M-Instruct with a LoRA adapter, as in section 3.
TRL adds the adapter, and its reference is the same model with the adapter
switched off. A fresh adapter starts at B = 0, so the policy equals the
reference and every loss is exactly log 2. To compare on a policy that
differs from the reference, we fill B with small random numbers. Both A and B are seeded (0).

    uv run python src/dpo_by_hand.py

Takes the GPU lock: it loads the model.
"""

from __future__ import annotations

import math

from common import OUTPUTS, Guard, free, get_device, load_hf_model, load_tokenizer, write_results

import torch
import torch.nn.functional as F

import data
from reward_head import chat_ids

BETA = 0.1
N_PAIRS = 8
TRL_BATCH = 2                     # pairs per TRL batch: 8 at once would need 2 GB of logits per copy
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
B_STD = 0.03                      # random LoRA B entries: moves each reply's log-prob by a few nats


def reply_logp(model, ids: list[int], start: int, device) -> float:
    """Sum of log p(token) over the reply's tokens: positions start .. len(ids)-1."""
    x = torch.tensor([ids], device=device)
    logits = model(input_ids=x).logits[0, :-1].float()      # position t predicts token t+1
    logp = F.log_softmax(logits, -1)
    targets = x[0, 1:]
    per_token = logp.gather(-1, targets[:, None])[:, 0]
    return float(per_token[start - 1:].sum())


def dpo_loss(pi_w: float, pi_l: float, ref_w: float, ref_l: float, beta: float = BETA) -> float:
    """-log sigmoid(beta * ((pi_w - ref_w) - (pi_l - ref_l))), from the four sums of log-probs."""
    z = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return math.log1p(math.exp(-z)) if z > -30 else -z


@torch.no_grad()
def by_hand(model, tok, pairs, device) -> list[dict]:
    rows = []
    for p in pairs:
        ids_w, s_w = chat_ids(tok, p["prompt"], p["chosen"])
        ids_l, s_l = chat_ids(tok, p["prompt"], p["rejected"])
        pi_w, pi_l = reply_logp(model, ids_w, s_w, device), reply_logp(model, ids_l, s_l, device)
        with model.disable_adapter():
            ref_w, ref_l = reply_logp(model, ids_w, s_w, device), reply_logp(model, ids_l, s_l, device)
        rows.append({"id": p["id"], "pi_chosen": pi_w, "pi_rejected": pi_l, "ref_chosen": ref_w,
                     "ref_rejected": ref_l, "reward_chosen": BETA * (pi_w - ref_w),
                     "reward_rejected": BETA * (pi_l - ref_l), "loss": dpo_loss(pi_w, pi_l, ref_w, ref_l),
                     "loss_policy_equals_reference": dpo_loss(ref_w, ref_l, ref_w, ref_l),
                     "reply_tokens": [len(ids_w) - s_w, len(ids_l) - s_l]})
    return rows


def make_trainer(model, tok, pairs):
    """TRL's DPOTrainer on the base model; TRL adds the LoRA adapter, and its reference is the adapter switched off."""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": p["prompt"]}],
                             "chosen": [{"role": "assistant", "content": p["chosen"]}],
                             "rejected": [{"role": "assistant", "content": p["rejected"]}]} for p in pairs])
    cfg = DPOConfig(output_dir=str(OUTPUTS / "dpo_check"), beta=BETA, per_device_eval_batch_size=TRL_BATCH,
                    max_length=data.MAX_TOKENS + 8, report_to=[], bf16=False, fp16=False,
                    disable_dropout=True, dataset_num_proc=None)
    return DPOTrainer(model=model, args=cfg, train_dataset=ds, eval_dataset=ds, processing_class=tok,
                      peft_config=LoraConfig(task_type="CAUSAL_LM", **LORA))


def with_trl(trainer) -> dict:
    trainer.model.eval()
    losses, n_completion = [], []
    with torch.no_grad():
        for batch in trainer.get_eval_dataloader():           # in order: no shuffling for eval
            losses.append(float(trainer.compute_loss(trainer.model, batch)))
            n_completion += batch["completion_mask"].sum(1).tolist()
    # Equal-sized batches, so the mean of batch means is the mean over pairs.
    m = {k: sum(v) / len(v) for k, v in trainer._metrics["eval"].items()}
    return {"loss": sum(losses) / len(losses), "batch_losses": losses,
            "rewards/chosen": m["rewards/chosen"], "rewards/rejected": m["rewards/rejected"],
            "rewards/accuracies": m["rewards/accuracies"], "rewards/margins": m["rewards/margins"],
            "logps/chosen": m["logps/chosen"], "logps/rejected": m["logps/rejected"],
            "completion_tokens": [int(x) for x in n_completion]}


def main() -> None:
    guard = Guard("dpo_by_hand")
    tok = load_tokenizer()
    pairs = data.splits(data.load_pairs(tok)[0])["heldout"][:N_PAIRS]
    device = get_device()
    torch.manual_seed(0)                                       # LoRA's A matrices draw from the global RNG
    trainer = make_trainer(load_hf_model(), tok, pairs)
    model = trainer.model
    g = torch.Generator().manual_seed(0)
    for name, prm in model.named_parameters():
        if "lora_B" in name:
            prm.data.copy_(torch.randn(prm.shape, generator=g) * B_STD)
    model.eval()
    guard.tick("load")

    rows = by_hand(model, tok, pairs, device)
    hand = {"loss": sum(r["loss"] for r in rows) / len(rows),
            "rewards/chosen": sum(r["reward_chosen"] for r in rows) / len(rows),
            "rewards/rejected": sum(r["reward_rejected"] for r in rows) / len(rows),
            "rewards/accuracies": sum(r["reward_chosen"] > r["reward_rejected"] for r in rows) / len(rows),
            "logps/chosen": sum(r["pi_chosen"] for r in rows) / len(rows),
            "logps/rejected": sum(r["pi_rejected"] for r in rows) / len(rows)}
    hand["rewards/margins"] = hand["rewards/chosen"] - hand["rewards/rejected"]
    guard.tick("by_hand")
    trl = with_trl(trainer)
    guard.tick("trl")
    diff = {k: abs(hand[k] - trl[k]) for k in hand}
    print(f"loss by hand {hand['loss']:.6f}, TRL {trl['loss']:.6f}, |diff| {diff['loss']:.2e}")
    for k in ("rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies", "logps/chosen"):
        print(f"  {k:<20} hand {hand[k]:+.5f}  TRL {trl[k]:+.5f}")
    print(f"  policy = reference: every loss is {rows[0]['loss_policy_equals_reference']:.6f} (log 2 = {math.log(2):.6f})")
    del model, trainer
    free()

    import peft
    import trl as trl_mod
    res = {"beta": BETA, "pairs": N_PAIRS, "lora": LORA, "lora_B_random_std": B_STD,
           "versions": {"trl": trl_mod.__version__, "peft": peft.__version__},
           "per_pair": rows, "hand": hand, "trl": trl, "abs_diff": diff,
           "log2": math.log(2), "guard": guard.report()}
    write_results("dpo_by_hand", res)


if __name__ == "__main__":
    main()
