"""Section 4: grokking on addition mod 97, with and without weight decay.

Task: given a and b in 0..96, predict (a + b) mod 97. There are 9,409 pairs;
a fixed 30% (2,823, split seed 0) is the training set and the other 6,586 are
the test set. The model is a one-layer transformer with no LayerNorm (d_model
128, 4 heads, MLP width 512, ReLU, learned positions); it reads the two
numbers as two tokens and answers at the last position. Full-batch AdamW,
rate 1e-3, betas (0.9, 0.98). One run per weight decay: 1.0 or 0.

The rule, fixed before the first run and not changed after:

* "Grokked" means test accuracy >= 95% at an evaluation step.
* Every run lasts 20,000 steps, whatever happens; no early stop.
* We evaluate every 10 steps up to step 500, then every 100 steps.
* We report two steps: the first evaluated step at which train accuracy is
  100%, and the first at which test accuracy is >= 95%, or "not within 20k"
  if that never happens.

Runs on the GPU (MPS) when there is one. At the end, one forward pass of the
final model on all 9,409 pairs on MPS and on the CPU checks that both agree.

    uv run python src/grok.py --wd 1.0 [--figures]     # writes results/grok_wd1.json
    uv run python src/grok.py --wd 0                   # writes results/grok_wd0.json
"""

from __future__ import annotations

import argparse
import math

from common import Guard, get_device, write_results

import torch

P = 97
TRAIN_FRAC = 0.3
SPLIT_SEED = 0
INIT_SEED = 0
STEPS = 20_000
LR = 1e-3
BETAS = (0.9, 0.98)
GROKKED = 0.95


def eval_steps(total: int = STEPS) -> list[int]:
    return sorted(set(range(0, 501, 10)) | set(range(500, total + 1, 100)))


def make_data(p: int = P, frac: float = TRAIN_FRAC, seed: int = SPLIT_SEED):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    x = torch.stack([a.flatten(), b.flatten()], dim=1)
    y = (x[:, 0] + x[:, 1]) % p
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    n = round(frac * len(x))
    return x[perm[:n]], y[perm[:n]], x[perm[n:]], y[perm[n:]]


class OneLayer(torch.nn.Module):
    """Embed, one attention block, one ReLU MLP, unembed. Residual stream, no LayerNorm."""

    def __init__(self, p: int = P, d: int = 128, heads: int = 4, d_mlp: int = 512, n_ctx: int = 2):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        s, sm = 1 / math.sqrt(d), 1 / math.sqrt(d_mlp)
        self.W_E = torch.nn.Parameter(torch.randn(p, d) * s)
        self.W_pos = torch.nn.Parameter(torch.randn(n_ctx, d) * s)
        self.W_QKV = torch.nn.Parameter(torch.randn(3, d, d) * s)
        self.W_O = torch.nn.Parameter(torch.randn(d, d) * s)
        self.W_in = torch.nn.Parameter(torch.randn(d, d_mlp) * s)
        self.b_in = torch.nn.Parameter(torch.zeros(d_mlp))
        self.W_out = torch.nn.Parameter(torch.randn(d_mlp, d) * sm)
        self.W_U = torch.nn.Parameter(torch.randn(d, p) * s)
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx, dtype=torch.bool)))

    def forward(self, x):
        h = self.W_E[x] + self.W_pos                                       # (B, T, d)
        B, T, d = h.shape
        q, k, v = (h @ self.W_QKV[i] for i in range(3))
        q, k, v = (t.view(B, T, self.heads, self.dh).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(-1)
        h = h + (att @ v).transpose(1, 2).reshape(B, T, d) @ self.W_O
        h = h + torch.relu(h @ self.W_in + self.b_in) @ self.W_out
        return h[:, -1] @ self.W_U                                          # logits at the last position


@torch.no_grad()
def measure(model, x, y):
    out = model(x)
    return torch.nn.functional.cross_entropy(out, y).item(), (out.argmax(-1) == y).float().mean().item()


def first_step(curve, key, threshold):
    for row in curve:
        if row[key] >= threshold:
            return row["step"]
    return None


def train(wd: float, device: torch.device, steps: int = STEPS, guard: Guard | None = None):
    x_tr, y_tr, x_te, y_te = (t.to(device) for t in make_data())
    torch.manual_seed(INIT_SEED)
    model = OneLayer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=wd)
    marks = set(eval_steps(steps))
    curve = []
    for step in range(steps + 1):
        if step in marks:
            tr_loss, tr_acc = measure(model, x_tr, y_tr)
            te_loss, te_acc = measure(model, x_te, y_te)
            wn = torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())).item()
            curve.append({"step": step, "train_loss": tr_loss, "train_acc": tr_acc,
                          "test_loss": te_loss, "test_acc": te_acc, "weight_norm": round(wn, 3)})
            if guard and step % 2000 == 0:
                guard.tick()
                print(f"  step {step:6d}  train {tr_acc:.3f}  test {te_acc:.3f}  |w| {wn:.1f}")
        if step == steps:
            break
        loss = torch.nn.functional.cross_entropy(model(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, curve, (len(x_tr), len(x_te))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    name = "grok_wd1" if args.wd == 1.0 else "grok_wd0" if args.wd == 0 else f"grok_wd{args.wd:g}"
    guard = Guard(name)
    device = get_device()
    model, curve, (n_tr, n_te) = train(args.wd, device, guard=guard)
    guard.tick("train")

    # MPS against CPU: the final model on every pair.
    x_all = torch.cat([t for t in make_data()[::2]])
    check = None
    if device.type == "mps":
        with torch.no_grad():
            a = model(x_all.to(device)).cpu()
            b = model.cpu()(x_all)
        check = {"pairs": len(x_all), "max_abs_diff_logits": float((a - b).abs().max()),
                 "logit_range": [round(float(b.min()), 2), round(float(b.max()), 2)],
                 "same_argmax_share": float((a.argmax(-1) == b.argmax(-1)).float().mean())}
        print(f"MPS vs CPU: max |diff| {check['max_abs_diff_logits']:.2e}, same answer on "
              f"{check['same_argmax_share']:.4%} of {check['pairs']} pairs")

    train_100 = first_step(curve, "train_acc", 1.0)
    test_95 = first_step(curve, "test_acc", GROKKED)
    for row in curve:
        for k in ("train_loss", "test_loss", "train_acc", "test_acc"):
            row[k] = round(row[k], 5)
    out = {"p": P, "train_frac": TRAIN_FRAC, "n_train": n_tr, "n_test": n_te, "split_seed": SPLIT_SEED,
           "init_seed": INIT_SEED, "weight_decay": args.wd, "lr": LR, "betas": list(BETAS), "steps": STEPS,
           "rule": {"grokked_test_acc": GROKKED, "cap_steps": STEPS, "early_stop": False,
                    "eval": "every 10 steps to 500, then every 100"},
           "model": {"layers": 1, "d_model": 128, "heads": 4, "d_mlp": 512, "layernorm": False,
                     "params": sum(p.numel() for p in model.parameters())},
           "device": device.type, "train_hits_100": train_100, "test_crosses_95": test_95,
           "grokked": test_95 is not None,
           "final": {k: curve[-1][k] for k in ("train_acc", "test_acc", "train_loss", "test_loss", "weight_norm")},
           "mps_vs_cpu": check, "curve": curve}
    print(f"wd {args.wd}: train hits 100% at step {train_100}; test crosses 95% at step "
          f"{test_95 if test_95 is not None else 'not within 20k'}")
    out["guard"] = guard.report()
    write_results(name, out)
    if args.figures:
        import figures
        figures.fig_grok()


if __name__ == "__main__":
    main()
