"""Section 1: the base model. Module 04's char-GPT, trained by plain next-character
loss on examples of 3-digit addition, and stopped early on purpose.

RL can only reinforce what the model already does sometimes, so the base must
be right often but not always. The training replies are 80% direct and 20%
worked; the loss covers reply characters only. Writes ``artifacts/base.pt``
(float16, the start of every RL run) and ``results/sft.json``.
"""

from __future__ import annotations

import argparse

import common  # noqa: F401  (laptop guard before torch)
import numpy as np
import torch
import torch.nn.functional as F

import rollout
import task
from common import ARTIFACTS, Guard, write_results

STEPS = 3000
BATCH = 256
LR = 1e-3
WORKED_SHARE = 0.2
SEED = 0
SEQ = task.PROMPT_LEN + task.MAX_REPLY      # 30


def batch(draw, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for a, b in draw(BATCH, rng):
        p = task.prompt(int(a), int(b))
        r = task.worked(a, b) if rng.random() < WORKED_SHARE else task.direct(a, b)
        ids = task.encode(p + r)
        ids = ids + [task.PAD] * (SEQ - len(ids))
        tgt = [-100] * (task.PROMPT_LEN - 1) + ids[task.PROMPT_LEN:]
        tgt = [t if t != task.PAD else -100 for t in tgt]
        xs.append(ids[:-1])
        ys.append(tgt)
    return torch.tensor(xs), torch.tensor(ys)


@torch.no_grad()
def evaluate(model, pairs: np.ndarray, samples: int = 8, seed: int = 1) -> dict:
    """Greedy accuracy, and accuracy of samples at temperature 1, by reply style."""
    P = rollout.prompt_ids(pairs)
    rg, _ = rollout.sample(model, P, greedy=True)
    r_g, t_g = rollout.rewards(rg, pairs)
    gen = torch.Generator().manual_seed(seed)
    rep = np.repeat(pairs, samples, 0)
    rs, _ = rollout.sample(model, P.repeat_interleave(samples, 0), gen=gen)
    r_s, t_s = rollout.rewards(rs, rep)
    w = np.array([task.is_worked(s) for s in t_s])
    r_s = r_s.numpy()
    return {
        "greedy_acc": float(r_g.mean()),
        "greedy_worked_share": float(np.mean([task.is_worked(s) for s in t_g])),
        "sample_acc": float(r_s.mean()),
        "sample_worked_share": float(w.mean()),
        "sample_acc_direct": float(r_s[~w].mean()) if (~w).any() else None,
        "sample_acc_worked": float(r_s[w].mean()) if w.any() else None,
        "sample_unfinished": float(np.mean([not s.endswith(".") for s in t_s])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--probe", action="store_true", help="print test numbers every 100 steps; save nothing")
    args = ap.parse_args()

    g = Guard("sft")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    test, draw = task.splits()
    model = rollout.new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    log, curve = [], []
    for step in range(1, args.steps + 1):
        model.train()
        x, y = batch(draw, rng)
        logits = model(x)[0]
        loss = F.cross_entropy(logits.reshape(-1, task.VOCAB), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            log.append({"step": step, "loss": round(loss.item(), 4)})
        if step % 250 == 0 and not args.probe:
            ev = evaluate(model, test[:200], samples=4)
            curve.append({"step": step, "greedy_acc": ev["greedy_acc"], "sample_acc": ev["sample_acc"],
                          "sample_acc_direct": ev["sample_acc_direct"], "sample_acc_worked": ev["sample_acc_worked"]})
        if args.probe and step % 100 == 0:
            print(step, round(loss.item(), 4), {k: round(v, 3) for k, v in evaluate(model, test[:200], samples=4).items()}, flush=True)
    g.tick("train")
    if args.probe:
        return

    # Round to float16 once: every RL run starts from exactly these weights.
    sd = {k: v.to(torch.float16) for k, v in model.state_dict().items()}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save(sd, ARTIFACTS / "base.pt")
    base = load_base()
    ev = evaluate(base, test, samples=8)
    g.tick("evaluate")
    print(ev)
    write_results("sft", {
        "config": {"steps": args.steps, "batch": BATCH, "lr": LR, "worked_share": WORKED_SHARE, "seed": SEED,
                   "params": base.num_params(), "n_layer": rollout.N_LAYER, "d_model": rollout.D_MODEL, "n_head": rollout.N_HEAD,
                   "test_pairs": len(test)},
        "loss_log": log,
        "curve_first_200_test": curve,
        "test": ev,
        "examples": {"prompt": task.prompt(374, 485), "direct": task.direct(374, 485), "worked": task.worked(374, 485)},
        "guard": g.report(),
    })


def load_base():
    model = rollout.new_model()
    sd = torch.load(ARTIFACTS / "base.pt")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model.eval()
    return model


if __name__ == "__main__":
    main()
