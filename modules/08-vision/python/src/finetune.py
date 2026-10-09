"""Fine-tune lab: adapt a pretrained ViT-tiny to the `beans` dataset.

`beans` is a tiny 3-class leaf-disease dataset (angular leaf spot, bean rust,
healthy). We take an ImageNet-pretrained ViT-tiny, throw away its 1000-class
head, bolt on a fresh 3-class head, and fine-tune the whole thing with the
Hugging Face ``Trainer`` on MPS. A few minutes of training turns a generic
feature extractor into a competent bean-leaf doctor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from utils import BEANS, get_device, vit_processor

MODEL_NAME = "WinKawaks/vit-tiny-patch16-224"


@dataclass
class FineTuneResult:
    model: object
    processor: object
    class_names: list[str]
    history: list[dict]
    test_acc: float
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    y_conf: np.ndarray = field(repr=False)


def _build_transform(processor):
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    size = processor.size["height"] if "height" in processor.size else processor.size["shortest_edge"]

    def transform(batch):
        pixel_values = []
        for img in batch["image"]:
            img = img.convert("RGB").resize((size, size))
            arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            arr = arr.permute(2, 0, 1)          # HWC -> CHW
            arr = (arr - mean) / std
            pixel_values.append(arr)
        return {"pixel_values": pixel_values, "labels": batch["labels"]}

    return transform


def _collate(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["labels"] for e in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def load_beans(processor):
    from datasets import load_dataset

    ds = load_dataset(BEANS)
    tf = _build_transform(processor)
    for split in ds:
        ds[split] = ds[split].with_transform(tf)
    class_names = load_dataset(BEANS, split="test").features["labels"].names
    return ds, class_names


def train_beans(epochs: int = 4, batch_size: int = 16, lr: float = 3e-4, seed: int = 0) -> FineTuneResult:
    import tempfile

    from sklearn.metrics import accuracy_score
    from transformers import Trainer, TrainingArguments, ViTForImageClassification

    torch.manual_seed(seed)
    processor = vit_processor(MODEL_NAME)
    ds, class_names = load_beans(processor)

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label={i: n for i, n in enumerate(class_names)},
        label2id={n: i for i, n in enumerate(class_names)},
        ignore_mismatched_sizes=True,   # replace the 1000-class head
    )

    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        return {"accuracy": accuracy_score(pred.label_ids, preds)}

    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
            seed=seed,
            dataloader_num_workers=0,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=_collate,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        history = [h for h in trainer.state.log_history]

        # honest test-set evaluation + per-example predictions for figures
        pred = trainer.predict(ds["test"])
        logits = torch.from_numpy(pred.predictions)
        probs = torch.softmax(logits, dim=-1)
        y_pred = probs.argmax(-1).numpy()
        y_conf = probs.max(-1).values.numpy()
        y_true = np.asarray(pred.label_ids)
        test_acc = float((y_pred == y_true).mean())

    model.eval().to(get_device())
    return FineTuneResult(
        model=model, processor=processor, class_names=class_names,
        history=history, test_acc=test_acc,
        y_true=y_true, y_pred=y_pred, y_conf=y_conf,
    )


if __name__ == "__main__":
    res = train_beans()
    print(f"\nbeans test accuracy: {res.test_acc:.4f}")
