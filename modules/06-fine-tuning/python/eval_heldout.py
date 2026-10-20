"""Held-out loss: does the adapter help on conversations it never saw?

The training loss in loss_history.json is measured on the same 512 examples
the model trains on (about three passes in the main lab), so it cannot tell
learning from memorising. This script scores the 119-conversation *test*
split of smoltalk everyday-conversations, which no run trains on, with the
base model and with each saved adapter.

Two numbers per model, both mean next-token cross-entropy (nats per token):
  all_tokens        every token of the chat-formatted conversation, the loss
                    the lab trains on (TRL does not mask the prompt here)
  assistant_tokens  only the assistant's replies, the part a chat model is
                    asked to produce

Needs the adapters from sft_train.py and solutions/ex_a_rank.py under
outputs/. Writes results/heldout.json and copies each run's
loss_history.json into results/ so the README's numbers sit in tracked files.

Run:  uv run python eval_heldout.py      (~2 min on an M5, ~3 GB peak)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import DATASET_CONFIG, DATASET_ID, MODEL_ID, pick_device

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "outputs"
RESULTS = HERE / "results"
RUNS = ["sft-lab", "ex-a-r4", "ex-a-r32"]


def encode(tok, messages):
    """Token ids for one conversation, and a mask marking assistant tokens.

    Each assistant reply spans from the end of the prompt that precedes it
    (with the generation prompt added) to the end of the conversation up to
    and including it.
    """
    ids = tok.apply_chat_template(messages, tokenize=True, return_dict=False)
    mask = [0] * len(ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(tok.apply_chat_template(messages[:i], tokenize=True,
                                            add_generation_prompt=True, return_dict=False))
        end = len(tok.apply_chat_template(messages[: i + 1], tokenize=True, return_dict=False))
        for j in range(start, end):
            mask[j] = 1
    return ids, mask


@torch.no_grad()
def score(model, data, device):
    """Summed loss and token counts over the split, all tokens and assistant only."""
    tot = {"all": [0.0, 0], "asst": [0.0, 0]}
    for ids, mask in data:
        x = torch.tensor([ids], device=device)
        logits = model(x).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none")
        m = torch.tensor(mask[1:], device=device, dtype=torch.bool)
        tot["all"][0] += nll.sum().item()
        tot["all"][1] += nll.numel()
        tot["asst"][0] += nll[m].sum().item()
        tot["asst"][1] += int(m.sum().item())
    return {
        "all_tokens": round(tot["all"][0] / tot["all"][1], 4),
        "assistant_tokens": round(tot["asst"][0] / tot["asst"][1], 4),
        "n_tokens": tot["all"][1],
        "n_assistant_tokens": tot["asst"][1],
    }


def main():
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    test = load_dataset(DATASET_ID, DATASET_CONFIG, split="test")
    data = [encode(tok, row["messages"]) for row in test]

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()
    res = {"model": MODEL_ID, "split": f"{DATASET_ID}/{DATASET_CONFIG} test",
           "n_conversations": len(data), "base": score(base, data, device)}
    print("base", res["base"])

    RESULTS.mkdir(exist_ok=True)
    for run in RUNS:
        adapter = OUT_ROOT / run / "adapter"
        if not adapter.exists():
            print(f"skip {run}: no adapter at {adapter}")
            continue
        model = PeftModel.from_pretrained(base, str(adapter)).eval()
        hist = json.loads((OUT_ROOT / run / "loss_history.json").read_text())
        res[run] = score(model, data, device) | {
            "final_train_loss": round(hist[-1]["loss"], 4),
            "train_steps": hist[-1]["step"],
        }
        print(run, res[run])
        base = model.unload()  # strip the adapter; the base weights were never changed
        shutil.copy(OUT_ROOT / run / "loss_history.json", RESULTS / f"{run}_loss_history.json")

    (RESULTS / "heldout.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {RESULTS / 'heldout.json'}")


if __name__ == "__main__":
    main()
