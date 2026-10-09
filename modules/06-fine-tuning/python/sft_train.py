"""Local SFT lab — LoRA fine-tune SmolLM2-360M-Instruct on everyday conversations.

This is the heart of the module. It:
  1. loads the base instruct model + tokenizer,
  2. loads a small slice of HuggingFaceTB/smoltalk (everyday-conversations),
  3. wraps the model with a LoRA adapter (peft),
  4. trains with TRL's SFTTrainer (report_to="trackio" in LOCAL mode),
  5. saves the adapter + the trainer's loss history to outputs/,
  6. plots its own loss-curve PNG (we don't screenshot the dashboard).

Everything is parameterised through `run_sft(...)` so the exercises can reuse it
(different rank, different target modules, your own dataset) without copy-paste.

Run the default lab:  uv run python sft_train.py
Open the dashboard:   uv run trackio show   (then visit the printed localhost URL)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.common import (
    DATASET_CONFIG,
    DATASET_ID,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGETS,
    MODEL_ID,
    TRAIN_SLICE,
    pick_device,
)

OUT_ROOT = Path(__file__).resolve().parent / "outputs"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def load_slice(n: int, seed: int = 0):
    """A few hundred everyday-conversation dialogues in chat (messages) format."""
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return ds


def run_sft(
    run_name: str = "sft-lab",
    rank: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    target_modules=LORA_TARGETS,
    max_steps: int = 200,
    train_slice: int = TRAIN_SLICE,
    learning_rate: float = 2e-4,
    dataset=None,
    report_to: str = "trackio",
    plot_title: str | None = None,
):
    """Fine-tune and return (loss_history, adapter_dir). Reused by the exercises."""
    device = pick_device()
    out_dir = OUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)

    lora = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )

    train_ds = dataset if dataset is not None else load_slice(train_slice)

    # fp32 on MPS: bf16/fp16 training paths on Apple's backend are still flaky,
    # and 360M in fp32 fits comfortably. On CUDA you'd flip bf16=True.
    cfg = SFTConfig(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch = 8
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        max_length=1024,
        packing=False,
        bf16=False,
        fp16=False,
        report_to=report_to,
        run_name=run_name,
        save_strategy="no",
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )

    # report the "you train X%" fact from the actual wrapped model
    trainer.model.print_trainable_parameters()

    trainer.train()

    # persist the adapter (small — a few MB) under outputs/ (gitignored)
    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # persist the loss history so we plot our OWN curve, not a dashboard screenshot
    hist = [
        {"step": h["step"], "loss": h["loss"]}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    (out_dir / "loss_history.json").write_text(json.dumps(hist, indent=2))

    if hist:
        plot_loss(hist, plot_title or f"LoRA SFT — {run_name}",
                  FIG_DIR / "loss_curve.png" if run_name == "sft-lab"
                  else out_dir / "loss_curve.png")

    return hist, adapter_dir


def plot_loss(hist, title, path):
    steps = [h["step"] for h in hist]
    loss = [h["loss"] for h in hist]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(steps, loss, color="#e11d48", lw=2, marker="o", ms=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss (cross-entropy)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if len(loss) > 1:
        ax.annotate(
            f"start {loss[0]:.2f} → end {loss[-1]:.2f}",
            xy=(steps[-1], loss[-1]),
            xytext=(0.55, 0.85), textcoords="axes fraction",
            fontsize=10, color="#334155",
        )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--slice", type=int, default=TRAIN_SLICE)
    ap.add_argument("--run-name", default="sft-lab")
    args = ap.parse_args()

    hist, adapter = run_sft(
        run_name=args.run_name, max_steps=args.steps, train_slice=args.slice
    )
    if hist:
        print(f"\nloss: {hist[0]['loss']:.4f} (step {hist[0]['step']}) "
              f"→ {hist[-1]['loss']:.4f} (step {hist[-1]['step']})")
    print(f"adapter saved to {adapter}")
