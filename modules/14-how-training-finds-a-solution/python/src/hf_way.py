"""Section 2, the library way: the same schedule and optimisers through
transformers.

1. ``get_cosine_schedule_with_warmup`` against our hand-written
   ``warmup_cosine``, at every one of the 1,570 steps.
2. ``Trainer`` with ``TrainingArguments(optim=...)`` on the same MLP and
   data, at each optimiser's best rate from optim_mlp.py, cosine schedule with
   10% warmup. Trainer shuffles batches its own way, so compare against the
   seed spread in results/optim_mlp.json, not against one run.

    uv run python src/hf_way.py
"""

from __future__ import annotations

import tempfile

from common import MLP, Guard, evaluate, load_fashion, read_results, write_results
from optim_mlp import BATCH, EPOCHS, THREADS, WARMUP_FRAC, warmup_cosine

import torch

# optim string and optim_args for each of our three optimisers. Trainer's
# "adamw_torch" with weight_decay=0 is plain Adam.
HF_OPTIM = {"sgd": ("sgd", None), "momentum": ("sgd", "momentum=0.9"), "adam": ("adamw_torch", None)}


class Rows(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "labels": self.y[i]}


class MLPForTrainer(MLP):
    """Trainer calls model(**batch) and reads the loss from the output."""

    def forward(self, x, labels=None):
        logits = self.net(x)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = torch.nn.functional.cross_entropy(logits, labels)
        return out


def schedule_check(total: int) -> dict:
    from transformers import get_cosine_schedule_with_warmup

    warmup = round(WARMUP_FRAC * total)
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total)
    diffs = []
    for step in range(total):
        diffs.append(abs(opt.param_groups[0]["lr"] - warmup_cosine(step, warmup, total)))
        opt.step()
        sched.step()
    return {"steps": total, "warmup": warmup, "max_abs_diff": max(diffs)}


def main() -> None:
    from transformers import Trainer, TrainingArguments

    torch.set_num_threads(THREADS)
    guard = Guard("hf_way")
    mlp = read_results("optim_mlp")
    check = schedule_check(mlp["steps"])
    print(f"schedule: max |HF - ours| over {check['steps']} steps = {check['max_abs_diff']}")

    x_tr, y_tr, x_te, y_te = load_fashion()
    runs = []
    for name, (optim, optim_args) in HF_OPTIM.items():
        lr = mlp["best"][f"{name}/warmup_cosine"]["lr"]
        torch.manual_seed(0)
        model = MLPForTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            targs = TrainingArguments(
                output_dir=tmp, optim=optim, optim_args=optim_args, learning_rate=lr, weight_decay=0.0,
                lr_scheduler_type="cosine", warmup_steps=WARMUP_FRAC, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, use_cpu=True, seed=0, report_to=[],
                save_strategy="no", logging_strategy="no", disable_tqdm=True, remove_unused_columns=False,
            )
            trainer = Trainer(model=model, args=targs, train_dataset=Rows(x_tr, y_tr))
            trainer.train()
            opt_cls = type(trainer.optimizer.optimizer if hasattr(trainer.optimizer, "optimizer") else trainer.optimizer).__name__
            steps = trainer.state.global_step
        model.eval()
        te_loss, te_acc = evaluate(model.net, x_te, y_te)
        seeds = mlp["best"][f"{name}/warmup_cosine"]["test_acc_seeds"]
        runs.append({"optimiser": name, "optim": optim, "optim_args": optim_args, "lr": lr, "torch_class": opt_cls,
                     "steps": steps, "test_acc": round(te_acc, 4), "test_loss": round(te_loss, 4),
                     "hand_loop_seeds": seeds})
        print(f"Trainer optim={optim!r} args={optim_args!r} lr {lr}: {opt_cls}, {steps} steps, test acc {te_acc:.4f}"
              f"  (hand loop, 3 seeds: {seeds})")
    guard.tick("trainer")

    out = {"schedule_check": check, "runs": runs, "epochs": EPOCHS, "batch": BATCH, "warmup_frac": WARMUP_FRAC}
    import transformers
    out["transformers"] = transformers.__version__
    out["guard"] = guard.report()
    write_results("hf_way", out)


if __name__ == "__main__":
    main()
