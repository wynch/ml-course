"""Train every model of the module, in chunks of under 5 minutes.

Main arms (sections 2 to 4), module 04's recipe: 5,000 AdamW steps of 64
windows of 128 characters on the full training split, rate 3e-4 with 150
warm-up steps and a cosine decay to 10%, weight decay 0.1, dropout 0.1.

    ar      causal, next-character loss                  (module 04's model, retrained here)
    mdm     bidirectional, masked diffusion loss
    mdm6x   the same at 30,000 steps: six times the compute (added after mdm's result)
    bd4     block diffusion, blocks of 4 characters
    bd16    block diffusion, blocks of 16 characters

Repeat arms (section 5): the same network on the first U characters of the
training split only, drawn again and again; constant rate 1e-3 after 100
warm-up steps, so one run can be scored at every epoch mark on the way.

    r100k_{ar,mdm,ard3}_s{0,1,2}   U = 100,000 characters, up to 200 epochs (ard3: AR, dropout 0.3)
    r100k_mdmlong_s0               the diffusion arm, seed 0, on to 800 epochs (added after the 200-epoch runs)
    r1m_{ar,mdm}_s0                U = the whole split (1,003,855), up to 64 epochs

An epoch is U characters drawn (as random windows), not one pass in order.
Every arm saves itself to artifacts/runs/<arm>.pt after each chunk and goes on
from there; no process trains for more than common.CHUNK_SECONDS.

    uv run python src/train.py --arm mdm          # one chunk of one arm
    uv run python src/train.py --main             # every chunk of the main arms, one process each
    uv run python src/train.py --repeat           # the same for the repeat arms
    uv run python src/train.py --export           # artifacts/models.pt: the main arms in 16 bits
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time

import common
import torch

import data as D
import diffusion as Df
from model import GPT, GPTConfig

BATCH = 64
EVAL_BLOCKS = 256
U_SMALL = 100_000
MARKS_SMALL = (1, 2, 4, 8, 16, 32, 64, 128, 200)
MARKS_FULL = (1, 2, 4, 8, 16, 32, 64)
MARKS_LONG = MARKS_SMALL + (400, 800)


def _arms() -> dict:
    arms = {
        "ar": {"obj": "ar", "steps": 5000, "sched": "cosine"},
        "mdm": {"obj": "mdm", "steps": 5000, "sched": "cosine"},
        "mdm6x": {"obj": "mdm", "steps": 30000, "sched": "cosine"},
        "bd4": {"obj": "bd", "L": 4, "steps": 5000, "sched": "cosine"},
        "bd16": {"obj": "bd", "L": 16, "steps": 5000, "sched": "cosine"},
    }
    for s in (0, 1, 2):
        for name, obj, drop in (("ar", "ar", 0.1), ("mdm", "mdm", 0.1), ("ard3", "ar", 0.3)):
            arms[f"r100k_{name}_s{s}"] = {"obj": obj, "U": U_SMALL, "marks": MARKS_SMALL, "dropout": drop,
                                          "seed": s, "sched": "constant"}
    arms["r100k_mdmlong_s0"] = {"obj": "mdm", "U": U_SMALL, "marks": MARKS_LONG, "seed": 0, "sched": "constant"}
    for name in ("ar", "mdm"):
        arms[f"r1m_{name}_s0"] = {"obj": name, "U": None, "marks": MARKS_FULL, "seed": 0, "sched": "constant"}
    return arms


ARMS = _arms()
MAIN = ("ar", "mdm", "bd4", "bd16", "mdm6x")
REPEAT = tuple(a for a in ARMS if a.startswith("r"))


def spec(arm: str, n_train: int) -> dict:
    """Fill in defaults; for repeat arms turn epoch marks into steps."""
    s = {"L": None, "dropout": 0.1, "seed": 1337, "lr": 3e-4, "warmup": 150, **ARMS[arm]}
    if s["sched"] == "constant":
        s["lr"], s["warmup"] = 1e-3, 100
        U = s["U"] or n_train
        s["U_chars"] = U
        s["mark_steps"] = {e: max(1, round(e * U / (BATCH * D.BLOCK))) for e in s["marks"]}
        s["steps"] = max(s["mark_steps"].values())
    return s


def lr_at(s: dict, it: int) -> float:
    if it < s["warmup"]:
        return s["lr"] * (it + 1) / s["warmup"]
    if s["sched"] == "constant":
        return s["lr"]
    frac = (it - s["warmup"]) / max(1, s["steps"] - s["warmup"])
    return s["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


def build(s: dict, vocab: int) -> GPT:
    return GPT(GPTConfig(vocab, dropout=s["dropout"], causal=s["obj"] == "ar"))


def loss_fn(model, s, x0, gen):
    if s["obj"] == "ar":
        return Df.ar_loss(model, x0)
    if s["obj"] == "mdm":
        return Df.mdm_loss(model, x0, Df.stratified_t(len(x0), gen, x0.device), gen)
    return Df.bd_loss(model, x0, s["L"], gen)


def score(model, s, blocks, lu) -> dict:
    """Held-out (or train) score per character, nats: exact for AR, the bound for diffusion."""
    if s["obj"] == "ar":
        return {"nll": round(Df.ar_nll(model, blocks, lu), 4)}
    out = {"elbo": round(Df.elbo(model, blocks, K=16, L=s["L"] if s["obj"] == "bd" else None), 4)}
    if s["obj"] == "mdm":
        out["l2r"] = round(Df.ordered_nll(model, blocks[:64], "l2r"), 4)
    return out


def train_chunk(arm: str, guard: common.Guard) -> bool:
    tok, train, val = D.load()
    s = spec(arm, len(train))
    data = train if s.get("U") is None else train[: s["U"]]
    dev = common.device()
    path = common.RUNS / f"{arm}.pt"
    torch.manual_seed(s["seed"])
    model = build(s, tok.vocab_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=0.1, betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(1000 + s["seed"])
    st = {"step": 0, "seconds": 0.0, "chunks": 0, "log": [], "marks": {}}
    if path.exists():
        ck = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        st = ck["progress"]
    if st["step"] >= s["steps"]:
        return True
    lu = D.log_unigram(train, tok.vocab_size)
    vb, tb = D.val_blocks(val, EVAL_BLOCKS), D.val_blocks(data, EVAL_BLOCKS)
    marks = {v: e for e, v in s.get("mark_steps", {}).items()}
    t0, acc = time.perf_counter(), st.get("acc", 0.0)
    model.train()
    while st["step"] < s["steps"] and time.perf_counter() - t0 < common.CHUNK_SECONDS:
        for g in opt.param_groups:
            g["lr"] = lr_at(s, st["step"])
        x0 = D.batch(data, BATCH, gen).to(dev)
        loss = loss_fn(model, s, x0, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        st["step"] += 1
        acc += loss.item()
        if st["step"] % 50 == 0:
            st["log"].append({"step": st["step"], "loss": round(acc / 50, 4)})
            acc = 0.0
        if st["step"] in marks:
            e = marks[st["step"]]
            st["marks"][str(e)] = {"step": st["step"], "val": score(model, s, vb, lu), "train": score(model, s, tb, lu)}
            print(f"  {arm} epoch {e}: {st['marks'][str(e)]}")
            model.train()
        guard.tick()
    st["acc"] = acc                   # the running sum toward the next 50-step average
    st["seconds"] = round(st["seconds"] + time.perf_counter() - t0, 1)
    st["chunks"] += 1
    common.RUNS.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": arm, "spec": s, "state": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                "torch_rng": torch.get_rng_state(), "progress": st}, path)
    print(f"  {arm}: step {st['step']}/{s['steps']}, {st['seconds']} s in {st['chunks']} chunks")
    return st["step"] >= s["steps"]


def finish(arm: str) -> dict:
    """Write results/runs/<arm>.json: the spec, the loss log, the epoch marks, the final scores."""
    tok, train, val = D.load()
    ck = torch.load(common.RUNS / f"{arm}.pt", weights_only=False, map_location="cpu")
    s = ck["spec"]
    model = build(s, tok.vocab_size)
    model.load_state_dict(ck["state"])
    model.to(common.device())
    out = {"arm": arm, "spec": {k: v for k, v in s.items() if k != "mark_steps"}, "params": model.num_params(),
           "progress": {k: ck["progress"][k] for k in ("step", "seconds", "chunks")},
           "loss_log": ck["progress"]["log"], "marks": ck["progress"]["marks"]}
    if arm in MAIN:
        lu = D.log_unigram(train, tok.vocab_size)
        out["val_final"] = score(model, s, D.val_blocks(val, EVAL_BLOCKS), lu)
    common.write_results(arm, out, common.RESULTS / "runs")
    return out


def load_model(arm: str, device: str | None = None) -> GPT:
    """A finished main arm, from artifacts/runs/ if present, else from the committed 16-bit export."""
    tok, _, _ = D.load()
    p = common.RUNS / f"{arm}.pt"
    if p.exists():
        ck = torch.load(p, weights_only=False, map_location="cpu")
        s, state = ck["spec"], ck["state"]
    else:
        ex = torch.load(common.ARTIFACTS / "models.pt", weights_only=False, map_location="cpu")[arm]
        s, state = ex["spec"], {k: v.float() for k, v in ex["state"].items()}
    m = build(s, tok.vocab_size)
    m.load_state_dict(state)
    return m.to(device or common.device()).eval()


def export() -> dict:
    out = {}
    for arm in MAIN:
        ck = torch.load(common.RUNS / f"{arm}.pt", weights_only=False, map_location="cpu")
        out[arm] = {"spec": ck["spec"], "state": {k: v.half() for k, v in ck["state"].items()}}
    path = common.ARTIFACTS / "models.pt"
    torch.save(out, path)
    print(f"  wrote {path}, {path.stat().st_size / 1e6:.1f} MB")
    return {"bytes": path.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--main", action="store_true")
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if a.main or a.repeat:
        for arm in (MAIN if a.main else REPEAT):
            while not (common.RESULTS / "runs" / f"{arm}.json").exists():
                subprocess.run([sys.executable, __file__, "--arm", arm], check=True)
        return
    if a.export:
        export()
        return
    if a.arm:
        guard = common.Guard(f"train {a.arm}", common.device())
        done = train_chunk(a.arm, guard)
        guard.tick("train")
        if done and not (common.RESULTS / "runs" / f"{a.arm}.json").exists():
            finish(a.arm)
            guard.tick("finish")
        common.log_guard(guard.report())


if __name__ == "__main__":
    main()
