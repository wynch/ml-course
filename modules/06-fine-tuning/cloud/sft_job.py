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
        --flavor t4-small \
        --timeout 2h \
        --secrets HF_TOKEN \
        cloud/sft_job.py

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
MAX_STEPS = int(os.environ.get("MAX_STEPS", "1000"))
TRAIN_SLICE = int(os.environ.get("TRAIN_SLICE", "8000"))

# Where to push the adapter when the job finishes (requires HF_TOKEN secret).
# Set to "" to skip pushing.
PUSH_REPO = os.environ.get("PUSH_REPO", "")  # e.g. "your-username/smollm-sft-lora"

# trackio Space for LIVE monitoring of the cloud run. Leave as-is to log to a
# free Space named "smollm-sft" under your account; set "" for local-only logs.
TRACKIO_SPACE = os.environ.get("TRACKIO_SPACE", "")  # e.g. "your-username/smollm-sft"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = torch.cuda.is_available()  # cloud GPUs (T4/A10G) do bf16/fp16 happily

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if bf16 else torch.float32
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
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        max_length=1024,
        packing=True,          # pack short convos → far better GPU utilisation
        bf16=bf16,
        report_to=report_to,
        run_name="sft-smollm-cloud",
        push_to_hub=bool(PUSH_REPO),
        hub_model_id=PUSH_REPO or None,
        save_strategy="steps" if PUSH_REPO else "no",
        save_steps=max(MAX_STEPS // 4, 1),
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
