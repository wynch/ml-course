"""Reference solution for exercise (c) — swap beans for Cat_and_Dog.

Demonstrates that the fine-tune recipe is dataset-agnostic. Cats and dogs are
heavily represented in ImageNet, so ViT-tiny transfers almost instantly: ~2
epochs on an 800-image subset reach about 96% test accuracy in under a minute on
an M-series Mac (results/swap_dataset.json).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# make src/ importable when run directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))

from finetune import _build_transform, _collate, MODEL_NAME
from utils import save_result, vit_processor

DATASET = "Bingsu/Cat_and_Dog"
N_TRAIN, N_VAL, N_TEST = 800, 200, 400


def build_splits(processor, seed: int = 0):
    from datasets import load_dataset

    full = load_dataset(DATASET)
    class_names = full["train"].features["labels"].names

    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(full["train"]), size=N_TRAIN + N_VAL, replace=False)
    train = full["train"].select(sorted(int(i) for i in tr_idx[:N_TRAIN]))
    val = full["train"].select(sorted(int(i) for i in tr_idx[N_TRAIN:]))
    te_idx = rng.choice(len(full["test"]), size=N_TEST, replace=False)
    test = full["test"].select(sorted(int(i) for i in te_idx))

    tf = _build_transform(processor)
    return train.with_transform(tf), val.with_transform(tf), test.with_transform(tf), class_names


def main(epochs: int = 2, seed: int = 0, max_steps: int | None = None, record: bool = False):
    import tempfile

    from sklearn.metrics import accuracy_score
    from transformers import Trainer, TrainingArguments, ViTForImageClassification

    torch.manual_seed(seed)
    processor = vit_processor(MODEL_NAME)
    train, val, test, class_names = build_splits(processor, seed)

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label={i: n for i, n in enumerate(class_names)},
        label2id={n: i for i, n in enumerate(class_names)},
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(pred):
        return {"accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1))}

    with tempfile.TemporaryDirectory() as tmp:
        kwargs = dict(
            output_dir=tmp, per_device_train_batch_size=16,
            per_device_eval_batch_size=16, num_train_epochs=epochs,
            learning_rate=3e-4, eval_strategy="epoch", save_strategy="no",
            logging_strategy="epoch", report_to="none",
            remove_unused_columns=False, seed=seed,
        )
        if max_steps is not None:      # used by the test to train just a few steps
            kwargs["max_steps"] = max_steps
        args = TrainingArguments(**kwargs)
        trainer = Trainer(model=model, args=args, train_dataset=train,
                          eval_dataset=val, data_collator=_collate,
                          compute_metrics=compute_metrics)
        trainer.train()
        val_acc = [h["eval_accuracy"] for h in trainer.state.log_history if "eval_accuracy" in h]
        runtime = next(h["train_runtime"] for h in trainer.state.log_history if "train_runtime" in h)
        test_metrics = trainer.evaluate(test)
    print(f"{DATASET}: test accuracy {test_metrics['eval_accuracy']:.4f}")
    if record:
        save_result("swap_dataset", {
            "what": f"{MODEL_NAME} on {DATASET}: {N_TRAIN} train, {N_VAL} val, {N_TEST} test, {epochs} epochs, seed {seed}",
            "val_acc_per_epoch": [round(a, 4) for a in val_acc],
            "test_acc": round(test_metrics["eval_accuracy"], 4),
            "train_runtime_s": round(runtime, 1),
        })
    return test_metrics["eval_accuracy"]


if __name__ == "__main__":
    main(record=True)
