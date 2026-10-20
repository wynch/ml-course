"""Sections 1 to 3: a char-GPT learns Shakespeare, then GSM8K's solutions,
four ways; every run logs what it learns, what it forgets, and how far it
moved from the start model.

    uv run python src/forget.py start      # repeat until it prints "start: done"
    uv run python src/forget.py run --all  # repeat until it prints "remaining: 0"

Each call stops after about four minutes and picks up where the last stopped.
Everything runs on the CPU, at 4 threads.
"""

from __future__ import annotations

import argparse
import math
import time

import common  # noqa: F401  (laptop guard before torch)
import torch
import torch.nn.functional as F

import data
from common import ARTIFACTS, RESULTS, Guard, write_results
from gpt import GPT, GPTConfig
from lora import add_lora

CHUNK_SECONDS = 240
RUNS = RESULTS / "runs"

# Section 1: the start model, module 04's settings (dropout 0 here).
START = dict(steps=5000, batch=64, lr=3e-4, warmup=150, weight_decay=0.1, seed=1337, eval_every=250)

# Sections 2 and 3: every arm trains on the new text for the same steps.
STEPS = 600
BATCH = 32
EVAL_EVERY = 25
WARMUP = 20
SEEDS = (0, 1, 2)
ARMS = {
    "full": dict(kind="full", lr=1e-3),
    "full_slow": dict(kind="full", lr=1e-4),
    "lora4": dict(kind="lora", lr=3e-3, r=4),
    "lora32": dict(kind="lora", lr=3e-3, r=32),
    "replay": dict(kind="full", lr=1e-3, replay=0.25),
    "distill": dict(kind="full", lr=1e-3, alpha=0.5),
}


def config() -> GPTConfig:
    return GPTConfig(vocab_size=len(data.vocab()), block_size=data.BLOCK, n_layer=3, n_head=4,
                     d_model=128, dropout=0.0)


def load_start() -> GPT:
    model = GPT(config())
    model.load_state_dict(torch.load(ARTIFACTS / "start.pt", weights_only=True))
    return model.eval()


# ------------------------------------------------------------------ measures

@torch.no_grad()
def logprobs(model: GPT, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return torch.cat([F.log_softmax(model(x[i:i + 32])[0], -1) for i in range(0, len(x), 32)])


def forward_kl(logp_ref: torch.Tensor, logp: torch.Tensor) -> float:
    """Mean over positions of KL(ref || model) = sum_c p_ref(c) (log p_ref(c) - log p(c)), in nats."""
    return float((logp_ref.exp() * (logp_ref - logp)).sum(-1).mean())


def cross_entropy(logp: torch.Tensor, y: torch.Tensor) -> float:
    return float(-logp.gather(-1, y.unsqueeze(-1)).mean())


def measure(model: GPT, ref: dict[str, torch.Tensor]) -> dict:
    """Loss on each held-out text, and forward KL to the start model on each."""
    out = {}
    for name, (x, y) in data.eval_windows().items():
        lp = logprobs(model, x)
        out[f"{name}_loss"] = round(cross_entropy(lp, y), 5)
        out[f"kl_{name}"] = round(forward_kl(ref[name], lp), 5)
    return out


@torch.no_grad()
def sample(model: GPT, prompt: str, n: int = 160, seed: int = 0) -> str:
    torch.manual_seed(seed)
    idx = data.encode(prompt).unsqueeze(0)
    out = model.generate(idx, n, temperature=0.8)
    return data.decode(out[0, len(prompt):])


PROMPTS = {"old": "ROMEO:\n", "new": "Q: Tom has 3 boxes"}


# ------------------------------------------------------------- section 1

def lr_start(it: int) -> float:
    if it < START["warmup"]:
        return START["lr"] * (it + 1) / START["warmup"]
    frac = (it - START["warmup"]) / max(1, START["steps"] - START["warmup"])
    return START["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


def train_start() -> None:
    g = Guard("forget.py start")
    ckpt = ARTIFACTS / "start_ckpt.pt"
    torch.manual_seed(START["seed"])
    model = GPT(config())
    opt = torch.optim.AdamW(model.parameters(), lr=START["lr"], betas=(0.9, 0.99),
                            weight_decay=START["weight_decay"])
    gen = torch.Generator().manual_seed(START["seed"])
    state = {"it": 0, "curve": [], "seconds": 0.0, "peak_rss_gb": 0.0}
    if (ARTIFACTS / "start.pt").exists() and (RESULTS / "start.json").exists():
        print("start: done")
        return
    if ckpt.exists():
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        gen.set_state(c["gen"]); state = c["state"]
    a = data.tensors()["a_train"]
    t0 = time.perf_counter()
    zero = {k: torch.zeros(1) for k in ("old", "new")}
    while state["it"] < START["steps"] and time.perf_counter() - t0 < CHUNK_SECONDS:
        it = state["it"]
        if it % START["eval_every"] == 0:
            m = measure(model, {k: logprobs(model, v[0]) for k, v in data.eval_windows().items()})
            state["curve"].append({"step": it, "old_loss": m["old_loss"], "new_loss": m["new_loss"]})
            print(f"  step {it}: old {m['old_loss']:.3f}, new {m['new_loss']:.3f}")
        model.train()
        for grp in opt.param_groups:
            grp["lr"] = lr_start(it)
        x, y = data.batch(a, START["batch"], gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        state["it"] += 1
    state["seconds"] += time.perf_counter() - t0
    state["peak_rss_gb"] = max(state["peak_rss_gb"], round(g.rss_peak_bytes() / 2**30, 2))
    if state["it"] < START["steps"]:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                    "state": state}, ckpt)
        print(f"start: step {state['it']} of {START['steps']}; run again")
        return
    torch.save(model.state_dict(), ARTIFACTS / "start.pt")
    ref = {k: logprobs(model, v[0]) for k, v in data.eval_windows().items()}
    final = measure(model, ref)
    state["curve"].append({"step": state["it"], "old_loss": final["old_loss"], "new_loss": final["new_loss"]})
    t = data.texts()
    write_results("start", {
        "settings": START | {"n_layer": 3, "n_head": 4, "d_model": 128, "block": data.BLOCK, "dropout": 0.0},
        "parameters": model.num_params(),
        "vocab_size": len(data.vocab()),
        "vocab_only_in_new": [c for c in data.vocab() if c not in set(t["a_train"] + t["a_val"])],
        "chars": {k: len(v) for k, v in t.items()},
        "eval_chars_per_text": data.N_EVAL_WINDOWS * data.BLOCK,
        "curve": state["curve"],
        "final": {"old_loss": final["old_loss"], "new_loss": final["new_loss"]},
        "samples": {k: sample(model, p) for k, p in PROMPTS.items()},
        "train_seconds": round(state["seconds"], 1),
        "guard": g.report() | {"peak_rss_gb_all_chunks": state["peak_rss_gb"],
                               "minutes_all_chunks": round(state["seconds"] / 60, 2)},
    })
    ckpt.unlink(missing_ok=True)
    print("start: done")


# ------------------------------------------------------------- sections 2 and 3

def soft_loss(logits: torch.Tensor, y: torch.Tensor, ref_logp: torch.Tensor, alpha: float) -> torch.Tensor:
    """Cross-entropy against (1 - alpha) * one-hot(true next char) + alpha * the start model's prediction."""
    logp = F.log_softmax(logits, -1)
    hard = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    soft = -(ref_logp.exp() * logp).sum(-1)
    return ((1 - alpha) * hard + alpha * soft).mean()


def train_run(arm: str, seed: int) -> dict:
    spec = ARMS[arm]
    g = Guard(f"forget.py run {arm} s{seed}")
    start = load_start()
    ref = {k: logprobs(start, v[0]) for k, v in data.eval_windows().items()}
    model = load_start()
    torch.manual_seed(seed)
    if spec["kind"] == "lora":
        params = add_lora(model, spec["r"])
    else:
        params = list(model.parameters())
    trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=spec["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    gen = torch.Generator().manual_seed(1000 + seed)
    t = data.tensors()
    n_old = round(BATCH * spec.get("replay", 0.0))
    log = []
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            m = measure(model, ref)
            log.append({"step": step} | m)
        if step == STEPS:
            break
        model.train()
        for grp in opt.param_groups:
            grp["lr"] = spec["lr"] * min(1.0, (step + 1) / WARMUP)
        x, y = data.batch(t["b_train"], BATCH - n_old, gen)
        if n_old:
            xo, yo = data.batch(t["a_train"], n_old, gen)
            x, y = torch.cat([x, xo]), torch.cat([y, yo])
        logits, loss = model(x, y)
        if "alpha" in spec:
            with torch.no_grad():
                ref_logp = F.log_softmax(start(x)[0], -1)
            loss = soft_loss(logits, y, ref_logp, spec["alpha"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    first, last = log[0], log[-1]
    return {
        "arm": arm, "seed": seed, "spec": spec, "steps": STEPS, "batch": BATCH,
        "trainable_parameters": trainable,
        "log": log,
        "final": {
            "old_loss": last["old_loss"], "new_loss": last["new_loss"],
            "forgetting": round(last["old_loss"] - first["old_loss"], 5),
            "new_gain": round(first["new_loss"] - last["new_loss"], 5),
            "kl_new": last["kl_new"], "kl_old": last["kl_old"],
        },
        "samples": {k: sample(model, p, seed=seed) for k, p in PROMPTS.items()},
        "guard": g.report(),
    }


def run_all() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [(a, s) for s in SEEDS for a in ARMS if not (RUNS / f"{a}_s{s}.json").exists()]
    t0 = time.perf_counter()
    while todo and time.perf_counter() - t0 < CHUNK_SECONDS - 60:
        arm, seed = todo.pop(0)
        print(f"run {arm} seed {seed}")
        write_results(f"{arm}_s{seed}", train_run(arm, seed), RUNS)
    print(f"remaining: {len(todo)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["start", "run"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "start":
        train_start()
    elif args.all:
        run_all()
    else:
        write_results(f"{args.arm}_s{args.seed}", train_run(args.arm, args.seed), RUNS)


if __name__ == "__main__":
    main()
