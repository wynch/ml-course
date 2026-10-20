# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers",
#     "trl",
#     "peft",
#     "datasets",
#     "torch",
#     "trackio",
# ]
# ///
"""Cloud SFT job — the local lab, sized up, as a self-contained uv script.

This is the SAME LoRA-SFT recipe as python/sft_train.py, but (a) packaged as a
PEP 723 inline-dependency script so `hf jobs uv run` can execute it with no repo
checkout, (b) pointed at a bigger model with more steps and more data, and
(c) wired to push the trained adapter to the Hub and stream live metrics to a
trackio Space.

  >>> IMPORTANT: this file is written to be RUN ON HUGGING FACE JOBS, NOT LOCALLY. <<<
  >>> Running it needs prepaid HF credits. It is entirely OPTIONAL. See README.md. <<<

Launch (from modules/06-fine-tuning/):

    hf jobs uv run \
        --name sft-smollm \
        --flavor a10g-small \
        --timeout 2h \
        --secrets HF_TOKEN \
        cloud/sft_job.py

Run the timing test in README.md first, read the seconds per step from
the logs, multiply by MAX_STEPS, and set --timeout to twice that.

Everything is overridable with env vars (MODEL_ID, MAX_STEPS, PUSH_REPO,
TRACKIO_SPACE) so you don't have to edit the file to retarget it.
"""

from __future__ import annotations

import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# --- knobs (env-overridable) ---------------------------------------------------
# Default to SmolLM3-3B — a real step up from the 360M local lab. Drop to
# HuggingFaceTB/SmolLM2-360M-Instruct for a cheap smoke test of the pipeline.
MODEL_ID = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolLM3-3B")
# everyday-conversations has 2,260 training rows, about 0.48M tokens: some 470
# packed 1,024-token sequences. At 16 sequences per step, 90 steps is about
# three passes, like the local lab.
MAX_STEPS = int(os.environ.get("MAX_STEPS", "90"))
TRAIN_SLICE = int(os.environ.get("TRAIN_SLICE", "2260"))  # the whole subset

# Where to push the adapter when the job finishes (requires HF_TOKEN secret).
# Set to "" to skip pushing.
PUSH_REPO = os.environ.get("PUSH_REPO", "")  # e.g. "your-username/smollm-sft-lora"

# trackio Space for LIVE monitoring of the cloud run. Empty (the default) means
# no Space and no trackio logging; set e.g. "your-username/smollm-sft" to watch.
TRACKIO_SPACE = os.environ.get("TRACKIO_SPACE", "")  # e.g. "your-username/smollm-sft"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # A10G and L4 (Ampere, Ada) have bf16; a T4 (Turing) has only fp16.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if bf16 else torch.float32
    )

    ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations",
                      split="train")
    ds = ds.shuffle(seed=0).select(range(min(TRAIN_SLICE, len(ds))))

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # trackio: passing trackio_space_id makes trackio create/use a HF Space so
    # you can watch loss curves live in the browser while the job runs.
    report_to = "trackio" if TRACKIO_SPACE else "none"

    cfg = SFTConfig(
        output_dir="/tmp/sft-out",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # 16 packed sequences per step
        gradient_checkpointing=True,    # recompute activations: fits 24 GB
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        max_length=1024,
        packing=True,          # pack short convos → far better GPU utilisation
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        run_name="sft-smollm-cloud",
        push_to_hub=bool(PUSH_REPO),
        hub_model_id=PUSH_REPO or None,
        save_strategy="steps" if PUSH_REPO else "no",
        save_steps=max(MAX_STEPS // 4, 1),
        # like the local lab, the loss covers every token, prompts included;
        # set assistant_only_loss=True (needs a template with generation
        # markers) to train on the replies alone
    )
    # trackio Space wiring (only read by the trackio callback when set)
    if TRACKIO_SPACE:
        cfg.trackio_space_id = TRACKIO_SPACE

    model.to(device)
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tok, peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    if PUSH_REPO:
        trainer.save_model("/tmp/sft-out/adapter")
        trainer.push_to_hub()
        print(f"pushed adapter to https://huggingface.co/{PUSH_REPO}")


if __name__ == "__main__":
    main()
