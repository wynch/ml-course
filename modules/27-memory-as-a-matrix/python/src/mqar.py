"""Sections 3 and 4: recall on MQAR against the size of the state, then one attention layer in four.

Every run trains one small model (width 64) from scratch on multi-query
associative recall (``models.MQAR``): sequences of 128 tokens holding 4 to 48
key-value pairs, then queries. Each training batch holds one pair count,
cycling through 4, 8, 16, 32, 48, after a curriculum of 1,000 steps on 4 and
8 pairs only. AdamW, peak rate 3e-3, 100 warm-up steps,
cosine to 10%, weight decay 0.1, batch 32, 4,000 steps. After training,
accuracy on 512 held-out sequences per pair count (the same sequences for
every run); every 250 steps, on the first 128 of them.

Runs are cached one JSON each in results/mqar/. A call trains pending runs
until its time budget is spent (default 4.5 minutes), saving a checkpoint in
python/.runs/ (not committed) when a run must stop part-way, and resumes it
next call. Repeat until it prints "pending: 0":

    uv run python src/mqar.py run
    uv run python src/mqar.py status
    uv run python src/mqar.py summarize --figures
"""

from __future__ import annotations

import argparse
import json
import time

import common
import torch

import models as M

VOCAB, T, D = 512, 128, 64
PAIRS = [4, 8, 16, 32, 48]
STEPS = 4000
LR = 3e-3
BATCH = 32
EVAL_SEQS = 512
CKPT = common.MODULE / "python" / ".runs"


WARM = 1000          # curriculum: the first WARM steps use only 4 and 8 pairs


def pairs_at(step: int) -> int:
    """Pair count of the batch at this step: 4 and 8 in turn for the first WARM steps, then all five."""
    if step < WARM:
        return PAIRS[step % 2]
    return PAIRS[step % len(PAIRS)]


def jobs() -> list[dict]:
    """The planned runs, in the order they train."""
    out = []
    for seed in (0, 1):
        for dk in (8, 16, 32):
            for rule in ("linear", "decay", "delta", "gated_delta"):
                out.append({"name": f"grid_{rule}_dk{dk}_s{seed}", "section": "grid",
                            "layers": [rule, rule], "d_head": dk, "seed": seed})
        out.append({"name": f"grid_attention_dk16_s{seed}", "section": "grid",
                    "layers": ["attention", "attention"], "d_head": 16, "seed": seed})
    hybrids = {"all_gated_delta": ["gated_delta"] * 4, "all_attention": ["attention"] * 4}
    for pos in (1, 3):                                   # second and last of the four layers
        hybrids[f"attention_at_{pos}"] = ["attention" if i == pos else "gated_delta" for i in range(4)]
    for seed in (0, 1):
        for label, layers in hybrids.items():
            out.append({"name": f"hybrid_{label}_s{seed}", "section": "hybrid", "label": label,
                        "layers": layers, "d_head": 8, "seed": seed})
    return out


def done(job) -> bool:
    return (common.MQAR_RUNS / f"{job['name']}.json").exists()


def eval_set():
    data = M.MQAR(VOCAB, T)
    gen = torch.Generator().manual_seed(12345)
    return {n: data.batch(n, EVAL_SEQS, gen) for n in PAIRS}


@torch.no_grad()
def evaluate(model, sets, device) -> dict:
    model.eval()
    out = {}
    for n, (x, y) in sets.items():
        accs = [M.mqar_accuracy(model, x[i:i + 128].to(device), y[i:i + 128].to(device))
                for i in range(0, len(x), 128)]
        out[str(n)] = round(sum(accs) / len(accs), 4)
    model.train()
    return out


def train(job, deadline: float, device) -> bool:
    """Train one run until it finishes or the deadline passes. True when finished."""
    torch.manual_seed(job["seed"])
    data = M.MQAR(VOCAB, T)
    model = M.RecallLM(VOCAB, D, job["d_head"], job["layers"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    gen = torch.Generator().manual_seed(1000 + job["seed"])
    ck = CKPT / f"{job['name']}.pt"
    state = {"step": 0, "curve": [], "train_seconds": 0.0, "chunks": 0}
    if ck.exists():
        saved = torch.load(ck, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        gen.set_state(saved["gen"])
        state = saved["state"]
    state["chunks"] += 1
    sets = eval_set()
    t0 = time.perf_counter()
    step = state["step"]
    while step < STEPS:
        n = pairs_at(step)
        x, y = data.batch(n, BATCH, gen)
        for g in opt.param_groups:
            g["lr"] = M.lr_at(step, STEPS, LR)
        loss = M.mqar_loss(model(x.to(device)), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 250 == 0 or step == STEPS:
            small = {n: (x[:128], y[:128]) for n, (x, y) in sets.items()}
            state["curve"].append({"step": step, "loss": round(loss.item(), 4), "acc": evaluate(model, small, device)})
            print(f"  {job['name']} step {step} loss {loss.item():.3f} acc {state['curve'][-1]['acc']}", flush=True)
        if time.perf_counter() > deadline and step < STEPS:
            break
    common.sync(device)
    state["train_seconds"] += time.perf_counter() - t0
    state["step"] = step
    if step < STEPS:
        CKPT.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(), "state": state}, ck)
        print(f"  {job['name']}: paused at step {step}", flush=True)
        return False
    acc = evaluate(model, sets, device)
    payload = dict(job, d_model=D, vocab=VOCAB, T=T, steps=STEPS, lr=LR, batch=BATCH, device=str(device),
                   params=M.n_params(model), state_floats=model.state_floats(T),
                   state_bytes_fp32=4 * model.state_floats(T), acc=acc, curve=state["curve"],
                   train_seconds=round(state["train_seconds"], 1), chunks=state["chunks"],
                   check_recurrent_vs_chunked=check_forms(model, sets, device))
    common.write_results(job["name"], payload, common.MQAR_RUNS)
    if ck.exists():
        ck.unlink()
    return True


@torch.no_grad()
def check_forms(model, sets, device) -> float | None:
    """The trained model's outputs, chunked form against the one-token loop (2 sequences)."""
    if all(k == "attention" for k in model.layers):
        return None
    x = sets[PAIRS[-1]][0][:2].to(device)
    model.eval()
    a, b = model(x), model(x, recurrent=True)
    model.train()
    return round((a - b).abs().max().item(), 7)


def run(minutes: float, only: str | None, dev: str | None = None) -> None:
    guard = common.Guard("mqar_chunk")
    deadline = time.perf_counter() + minutes * 60
    device = common.get_device(dev)
    finished = []
    for job in jobs():
        if done(job) or (only and not all(part in job["name"] for part in only.split(","))):
            continue
        if time.perf_counter() > deadline:
            break
        print(f"run {job['name']}", flush=True)
        if train(job, deadline, device):
            finished.append(job["name"])
        guard.tick()
    rep = guard.report()
    log = common.MQAR_RUNS / "chunks.jsonl"
    with log.open("a") as f:
        f.write(json.dumps({"finished": finished, "guard": rep}) + "\n")
    status()


def status() -> None:
    js = jobs()
    pend = [j["name"] for j in js if not done(j)]
    print(f"done: {len(js) - len(pend)}, pending: {len(pend)}")


def steps_to_learn(run) -> int | None:
    """First logged step at which half the queries with 4 pairs are right (None: never)."""
    return next((c["step"] for c in run["curve"] if c["acc"]["4"] >= 0.5), None)


def summarize(figures: bool) -> None:
    runs = [json.loads((common.MQAR_RUNS / f"{j['name']}.json").read_text()) for j in jobs() if done(j)]
    chunks = [json.loads(l) for l in (common.MQAR_RUNS / "chunks.jsonl").read_text().splitlines()]
    grid, hybrid, check = {}, {}, {}
    for r in runs:
        if r["section"] == "grid":
            key = f"{r['layers'][0]}_dk{r['d_head']}"
            g = grid.setdefault(key, {"rule": r["layers"][0], "d_head": r["d_head"], "params": r["params"],
                                      "state_bytes": r["state_bytes_fp32"], "seeds": {}})
            g["seeds"][str(r["seed"])] = r["acc"]
            g.setdefault("steps_to_learn", {})[str(r["seed"])] = steps_to_learn(r)
        elif r["section"] == "check":
            c = check.setdefault(f"{r['layers'][0]}_dk{r['d_head']}_alpha_bias{r['alpha_bias']}",
                                 {"rule": r["layers"][0], "d_head": r["d_head"], "alpha_bias": r["alpha_bias"], "seeds": {}})
            c["seeds"][str(r["seed"])] = r["acc"]
        else:
            h = hybrid.setdefault(r["label"], {"layers": r["layers"], "params": r["params"],
                                               "state_bytes": r["state_bytes_fp32"], "seeds": {}})
            h["seeds"][str(r["seed"])] = r["acc"]
            h.setdefault("steps_to_learn", {})[str(r["seed"])] = steps_to_learn(r)
    for g in list(grid.values()) + list(hybrid.values()) + list(check.values()):
        accs = list(g["seeds"].values())
        g["mean"] = {str(n): round(sum(a[str(n)] for a in accs) / len(accs), 4) for n in PAIRS}
        g["spread"] = {str(n): round(max(a[str(n)] for a in accs) - min(a[str(n)] for a in accs), 4) for n in PAIRS}
        # a seed "learned" when it reads at least half the queries right with 4 pairs; below that it never left chance
        g["learned_seeds"] = sum(a["4"] >= 0.5 for a in accs)
    worst_form = max((r["check_recurrent_vs_chunked"] or 0) for r in runs)
    out = {"vocab": VOCAB, "T": T, "d_model": D, "pairs": PAIRS, "steps": STEPS, "lr": LR, "batch": BATCH,
           "eval_seqs": EVAL_SEQS, "runs": len(runs), "grid": grid, "hybrid": hybrid, "check": check,
           "max_recurrent_vs_chunked_trained": worst_form,
           "train_minutes_total": round(sum(r["train_seconds"] for r in runs) / 60, 1),
           "chunk_count": len(chunks),
           "chunk_max_minutes": max(c["guard"]["minutes"] for c in chunks),
           "chunk_peak_rss_gb": max(c["guard"]["peak_rss_gb"] for c in chunks),
           "chunk_peak_mps_gb": max(c["guard"]["peak_mps_driver_gb"] for c in chunks),
           "guard": chunks[-1]["guard"]}
    common.write_results("mqar", out)
    if figures:
        import figures as F

        F.fig_mqar()
        F.fig_hybrid()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "status", "summarize"])
    ap.add_argument("--minutes", type=float, default=4.5)
    ap.add_argument("--only", default=None, help="train only runs whose name contains each comma-separated part")
    ap.add_argument("--device", default=None, help="cpu or mps; default mps when present")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.what == "run":
        run(args.minutes, args.only, args.device)
    elif args.what == "status":
        status()
    else:
        summarize(args.figures)


if __name__ == "__main__":
    main()
