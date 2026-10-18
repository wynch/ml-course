"""Section 1: distil module 04's char-GPT into a half-size student.

Teacher: module 04's trained char-GPT (3 layers, d_model 128, 619,776
parameters). Student: the same class at d_model 88 (299,376 parameters,
48% of the teacher). Two students per seed, trained under identical
conditions: same initial weights, same batches, same steps, same optimiser.
Only the loss differs:

    scratch:  CE(student logits, next char)
    distil:   (1 - ALPHA) * CE  +  ALPHA * T^2 * KL(teacher_T || student_T)

where p_T = softmax(logits / T). The T^2 keeps the soft term's gradient on
the same scale as CE's (Hinton et al., 2015).

Every loss in results/distil.json is plain next-character cross-entropy on
module 04's validation split (the last 10% of tiny-shakespeare), which the
teacher never trained on.

Run me:
    uv run python src/distil.py            # numbers, results/distil.json
    uv run python src/distil.py --figures  # also figures/distil.png
"""

from __future__ import annotations

import argparse

import common  # noqa: F401  (MPS guard before torch)
import torch
import torch.nn.functional as F

from common import Guard, get_device, seed_all, write_results
from tiny_gpt import load_tiny_gpt, new_gpt, splits

STUDENT_D = 88
STEPS = 1000
BATCH = 64
BLOCK = 128
LR = 1e-3
WARMUP = 50
WEIGHT_DECAY = 0.1
T = 2.0
ALPHA = 0.5
SEEDS = (0, 1, 2)
EVAL_EVERY = 100


def distil_loss(student_logits, teacher_logits, targets, T: float = T, alpha: float = ALPHA) -> torch.Tensor:
    """(1 - alpha) * CE(student, targets) + alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))."""
    V = student_logits.size(-1)
    s = student_logits.reshape(-1, V)
    t = teacher_logits.reshape(-1, V)
    ce = F.cross_entropy(s, targets.reshape(-1))
    kl = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction="batchmean")
    return (1 - alpha) * ce + alpha * T * T * kl


def lr_at(step: int) -> float:
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR * 0.5 * (1 + torch.cos(torch.tensor(p * torch.pi)).item())


@torch.no_grad()
def val_loss(model, val: torch.Tensor, device) -> float:
    """Mean next-char CE over every position of the validation split, in windows of BLOCK + 1."""
    n = (len(val) - 1) // BLOCK
    x = val[: n * BLOCK].view(n, BLOCK)
    y = val[1 : n * BLOCK + 1].view(n, BLOCK)
    model.eval()
    total = 0.0
    for i in range(0, n, 256):
        logits, _ = model(x[i : i + 256].to(device))
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[i : i + 256].reshape(-1).to(device),
                                       reduction="sum"))
    return total / (n * BLOCK)


def batch_starts(seed: int, n_train: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randint(n_train - BLOCK - 1, (STEPS, BATCH), generator=g)


def train(kind: str, seed: int, teacher, train_data, val, device) -> dict:
    seed_all(seed)
    student = new_gpt(STUDENT_D).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    starts = batch_starts(seed, len(train_data))
    offs = torch.arange(BLOCK + 1)
    curve = [{"step": 0, "val_loss": round(val_loss(student, val, device), 4)}]
    for step in range(STEPS):
        student.train()
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        chunk = train_data[starts[step][:, None] + offs].to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits, _ = student(x)
        if kind == "scratch":
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            with torch.no_grad():
                t_logits, _ = teacher(x)
            loss = distil_loss(logits, t_logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if (step + 1) % EVAL_EVERY == 0:
            curve.append({"step": step + 1, "val_loss": round(val_loss(student, val, device), 4)})
    return {"kind": kind, "seed": seed, "val_loss": curve[-1]["val_loss"], "curve": curve,
            "params": sum(p.numel() for p in student.parameters())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    guard = Guard("distil.py")
    device = get_device()
    teacher, tok = load_tiny_gpt(str(device))
    for p in teacher.parameters():
        p.requires_grad_(False)
    train_data, val = splits(tok)
    teacher_loss = round(val_loss(teacher, val, device), 4)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"teacher: {teacher_params:,} params, val loss {teacher_loss}")
    guard.tick("teacher")

    runs = []
    for seed in SEEDS:
        for kind in ("scratch", "distil"):
            r = train(kind, seed, teacher, train_data, val, device)
            runs.append(r)
            print(f"  seed {seed} {kind:8s} val loss {r['val_loss']:.4f}")
            guard.tick(f"{kind}_seed{seed}")

    def mean(kind):
        v = [r["val_loss"] for r in runs if r["kind"] == kind]
        return round(sum(v) / len(v), 4)

    gaps = [round(next(r["val_loss"] for r in runs if r["kind"] == "scratch" and r["seed"] == s)
                  - next(r["val_loss"] for r in runs if r["kind"] == "distil" and r["seed"] == s), 4) for s in SEEDS]
    out = {
        "teacher": {"params": teacher_params, "d_model": 128, "n_layer": 3, "val_loss": teacher_loss,
                    "trained_steps": 5000, "trained_batch": 64},
        "student": {"params": runs[0]["params"], "d_model": STUDENT_D, "n_layer": 3, "n_head": 4,
                    "share_of_teacher": round(runs[0]["params"] / teacher_params, 3)},
        "setup": {"steps": STEPS, "batch": BATCH, "block": BLOCK, "lr": LR, "warmup": WARMUP, "schedule": "cosine",
                  "weight_decay": WEIGHT_DECAY, "T": T, "alpha": ALPHA, "seeds": list(SEEDS),
                  "val_chars": len(val), "device": str(device)},
        "runs": runs,
        "mean_val_loss": {"scratch": mean("scratch"), "distil": mean("distil")},
        "gap_scratch_minus_distil": gaps,
    }
    print(f"mean val loss: scratch {out['mean_val_loss']['scratch']}, distil {out['mean_val_loss']['distil']}; "
          f"gaps per seed {gaps}")
    out["guard"] = guard.report()
    write_results("distil", out)
    if args.figures:
        import figures

        figures.distil()


if __name__ == "__main__":
    main()
