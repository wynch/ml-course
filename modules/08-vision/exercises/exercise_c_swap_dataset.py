"""Exercise (c) — swap beans for another small HF image dataset.

The fine-tune machinery in ``src/finetune.py`` is dataset-agnostic: it only
needs images and integer labels. Prove it by pointing the same ViT-tiny at a
*different* small dataset — here `Bingsu/Cat_and_Dog` (2 classes, real photos) —
and report what transfers.

Because ImageNet already contains many cat and dog breeds, you should expect
this to converge even faster and higher than beans. Cat_and_Dog has no
validation split and is larger than beans, so we subsample and carve a
validation set out of train.

    uv run python ../exercises/exercise_c_swap_dataset.py

Fill in the marked lines. Expected: >97% val accuracy in well under a minute on
an M-series Mac with the small subset below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# make src/ importable when run directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))

# Reuse everything from the beans lab.
from finetune import _build_transform, _collate, MODEL_NAME
from utils import vit_processor

DATASET = "Bingsu/Cat_and_Dog"
N_TRAIN, N_VAL, N_TEST = 800, 200, 400   # keep it a few-minutes job


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


def main(epochs: int = 2, seed: int = 0):
    import tempfile

    from sklearn.metrics import accuracy_score
    from transformers import Trainer, TrainingArguments, ViTForImageClassification

    torch.manual_seed(seed)
    processor = vit_processor(MODEL_NAME)
    train, val, test, class_names = build_splits(processor, seed)

    # TODO(you): build the ViT-tiny classifier with a fresh head for
    #   len(class_names) classes (see finetune.train_beans for the pattern:
    #   num_labels, id2label, label2id, ignore_mismatched_sizes=True).
    model = ...  # replace me
    raise NotImplementedError

    def compute_metrics(pred):
        return {"accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1))}

    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp, per_device_train_batch_size=16,
            per_device_eval_batch_size=16, num_train_epochs=epochs,
            learning_rate=3e-4, eval_strategy="epoch", save_strategy="no",
            logging_strategy="epoch", report_to="none",
            remove_unused_columns=False, seed=seed,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train,
                          eval_dataset=val, data_collator=_collate,
                          compute_metrics=compute_metrics)
        trainer.train()
        test_metrics = trainer.evaluate(test)
    print(f"{DATASET}: test accuracy {test_metrics['eval_accuracy']:.4f}")
    return test_metrics["eval_accuracy"]


if __name__ == "__main__":
    main()
