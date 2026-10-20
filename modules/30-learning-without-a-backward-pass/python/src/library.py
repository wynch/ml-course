"""Section 4, the library way: MeZO against LoRA backprop on a real language model.

    uv run python src/library.py base             # the untrained model on SST-2 validation
    uv run python src/library.py lora             # LoRA backprop, one chunk; run until "remaining": 0
    uv run python src/library.py mezo             # MeZO on every weight, one chunk; run until "remaining": 0
    uv run python src/library.py summary          # results/library.json

SmolLM2-360M-Instruct (module 17's pin) reads "<sentence> It was" and we
compare its next-token scores for " great" and " terrible": a two-way
classifier made from the language model's own head, MeZO's prompt for SST-2.
Training uses 1,000 training sentences (500 per label, seed 0), batches of
16, loss the two-way cross-entropy. Validation is all 872 sentences of the
SST-2 validation split. Both arms run in float32 on MPS (MeZO's tiny
nudges vanish in bfloat16).

MeZO stores no weights between chunks. Its whole training run is a list of
(seed, projected gradient) pairs in results/library/mezo_steps.jsonl: a new
chunk rebuilds the model by replaying them, one nudge per step, no forward
pass. LoRA saves its adapter and optimizer to python/.runs/.
"""

from __future__ import annotations

import json
import sys
import time

import common
import torch

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"     # module 17's pin
SST2_ID = "stanfordnlp/sst2"
SST2_REVISION = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"         # the course's pin
WORDS = (" terrible", " great")                                     # label 0, label 1
TEMPLATE = "{s} It was"
BATCH = 16
N_TRAIN = 1000
LORA = {"steps": 300, "lr": 3e-4, "micro_batches": 2, "r": 8, "alpha": 16, "targets": ["q_proj", "v_proj"], "eval_every": 50}
MEZO = {"steps": 2000, "lr": 1e-5, "eps": 1e-3, "eval_every": 250}
CHUNK_SECONDS = 240
OUT = common.RESULTS / "library"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def data():
    from datasets import load_dataset

    ds = load_dataset(SST2_ID, revision=SST2_REVISION)
    tr = ds["train"]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(tr), generator=g).tolist()
    pick = {0: [], 1: []}
    for i in perm:
        lab = tr[i]["label"]
        if len(pick[lab]) < N_TRAIN // 2:
            pick[lab].append(i)
        if all(len(v) == N_TRAIN // 2 for v in pick.values()):
            break
    idx = sorted(pick[0] + pick[1])
    train = [(tr[i]["sentence"].strip(), tr[i]["label"]) for i in idx]
    val = [(r["sentence"].strip(), r["label"]) for r in ds["validation"]]
    return train, val


def load(tokenizer_only: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer_only:
        return tok, None
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32).to(DEVICE)
    model.eval()                                    # no dropout in either arm
    return tok, model


def word_ids(tok) -> list[int]:
    ids = [tok.encode(w) for w in WORDS]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]


def two_way_logits(model, tok, sentences: list[str]) -> torch.Tensor:
    """Scores for (" terrible", " great") after each prompt, shape (n, 2).
    Left padding, positions counted from each prompt's first real token, and
    logits for the last position only (the full vocabulary at every position
    would be the largest tensor in the step)."""
    tok.padding_side = "left"
    enc = tok([TEMPLATE.format(s=s) for s in sentences], return_tensors="pt", padding=True).to(DEVICE)
    pos = (enc.attention_mask.cumsum(1) - 1).clamp_min(0)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pos, logits_to_keep=1).logits
    return out[:, -1, word_ids(tok)]


def batch_loss(model, tok, batch) -> torch.Tensor:
    y = torch.tensor([b[1] for b in batch], device=DEVICE)
    return torch.nn.functional.cross_entropy(two_way_logits(model, tok, [b[0] for b in batch]).float(), y)


@torch.no_grad()
def evaluate(model, tok, val, bs: int = 32) -> float:
    right = 0
    for i in range(0, len(val), bs):
        chunk = val[i:i + bs]
        pred = two_way_logits(model, tok, [c[0] for c in chunk]).argmax(1).cpu()
        right += int((pred == torch.tensor([c[1] for c in chunk])).sum())
    return right / len(val)


def batches(train, step: int):
    """Batch for a step: epoch-wise shuffles from a fixed seed, so both arms see the same order."""
    per = len(train) // BATCH
    epoch, k = divmod(step, per)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(500 + epoch)).tolist()
    return [train[i] for i in perm[k * BATCH:(k + 1) * BATCH]]


class Mem:
    """Peaks of MPS memory, polled after each step: what live tensors hold
    (current_allocated) and what the Metal driver holds for the process."""

    def __init__(self):
        self.live = self.driver = 0

    def poll(self):
        if DEVICE.type == "mps":
            self.live = max(self.live, torch.mps.current_allocated_memory())
            self.driver = max(self.driver, torch.mps.driver_allocated_memory())

    def report(self):
        return {"mps_live_peak_gb": round(self.live / 2**30, 3), "mps_driver_peak_gb": round(self.driver / 2**30, 3)}


def release() -> None:
    """Hand cached MPS blocks back after each step, so the driver's pool
    tracks what the step needs instead of its own high-water mark."""
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def log_chunk(arm: str, guard, mem, extra: dict) -> None:
    g = guard.report()
    g.update(mem.report())
    g.update(extra)
    g["arm"] = arm
    with open(OUT / "chunks.jsonl", "a") as f:
        f.write(json.dumps(g) + "\n")


# ------------------------------------------------------------------- base
def base() -> None:
    guard = common.Guard("library base", "mps")
    tok, model = load()
    _, val = data()
    mem = Mem()
    acc = evaluate(model, tok, val)
    mem.poll()
    OUT.mkdir(parents=True, exist_ok=True)
    common.write_results("base", {"model": MODEL_ID, "revision": MODEL_REVISION, "val_acc": round(acc, 4),
                                  "n_val": len(val), "params": sum(p.numel() for p in model.parameters()),
                                  "memory": mem.report(), "guard": guard.report()}, OUT)
    print({"val_acc": acc})


# ------------------------------------------------------------------- LoRA
def lora() -> None:
    from peft import LoraConfig, get_peft_model

    guard = common.Guard("library lora", "mps")
    OUT.mkdir(parents=True, exist_ok=True)
    tok, model = load()
    model = get_peft_model(model, LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=0.0,
                                             target_modules=LORA["targets"]))
    train, val = data()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LORA["lr"], weight_decay=0.0)
    state = common.CACHE / "lora.pt"
    log = read_jsonl(OUT / "lora_log.jsonl")
    step = 0
    if state.exists():
        st = torch.load(state, weights_only=False)
        for p, q in zip(params, st["params"]):
            p.data.copy_(q.to(DEVICE))
        opt.load_state_dict(st["opt"])
        step = st["step"]
    mem = Mem()
    t0 = time.perf_counter()
    while step < LORA["steps"] and time.perf_counter() - guard.t0 < CHUNK_SECONDS:
        batch = batches(train, step)
        half = BATCH // LORA["micro_batches"]
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for k in range(0, BATCH, half):                 # two halves of 8: same gradient, half the activations
            part = batch_loss(model, tok, batch[k:k + half]) / LORA["micro_batches"]
            mem.poll()                                  # activations of this half still alive
            part.backward()
            loss += float(part)
        opt.step()
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        mem.poll()
        release()
        step += 1
        row = {"step": step, "loss": round(loss, 5)}
        if step % LORA["eval_every"] == 0:
            row["val_acc"] = round(evaluate(model, tok, val), 4)
            print(row, flush=True)
        log.append(row)
        with open(OUT / "lora_log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    secs = time.perf_counter() - t0
    common.CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"params": [p.detach().cpu() for p in params], "opt": opt.state_dict(), "step": step}, state)
    n_train = sum(p.numel() for p in params)
    log_chunk("lora", guard, mem, {"steps_done": step, "train_seconds": round(secs, 1), "trainable_params": n_train})
    print(json.dumps({"remaining": LORA["steps"] - step}))


# ------------------------------------------------------------------- MeZO
@torch.no_grad()
def nudge(params, seed: int, scale: float) -> None:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for p in params:
        p.add_(torch.randn(p.shape, generator=g, device=DEVICE), alpha=scale)


def mezo() -> None:
    guard = common.Guard("library mezo", "mps")
    OUT.mkdir(parents=True, exist_ok=True)
    tok, model = load()
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(model.parameters())
    train, val = data()
    path = OUT / "mezo_steps.jsonl"
    steps = read_jsonl(path)
    t_replay = time.perf_counter()
    for s in steps:                                   # rebuild the trained weights from seeds alone
        nudge(params, s["seed"], -MEZO["lr"] * s["g"])
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    replay = time.perf_counter() - t_replay
    mem = Mem()
    seeds = torch.Generator().manual_seed(7)
    for _ in steps:
        torch.randint(2**62, (1,), generator=seeds)
    step = len(steps)
    t0 = time.perf_counter()
    with torch.no_grad():
        while step < MEZO["steps"] and time.perf_counter() - guard.t0 < CHUNK_SECONDS:
            seed = int(torch.randint(2**62, (1,), generator=seeds))
            batch = batches(train, step)
            nudge(params, seed, +MEZO["eps"])
            lp = float(batch_loss(model, tok, batch))
            mem.poll()
            nudge(params, seed, -2 * MEZO["eps"])
            lm = float(batch_loss(model, tok, batch))
            nudge(params, seed, +MEZO["eps"])
            g = (lp - lm) / (2 * MEZO["eps"])
            nudge(params, seed, -MEZO["lr"] * g)
            mem.poll()
            release()
            step += 1
            row = {"step": step, "seed": seed, "g": g, "loss": round((lp + lm) / 2, 5)}
            if step % MEZO["eval_every"] == 0:
                row["val_acc"] = round(evaluate(model, tok, val), 4)
                print({k: row[k] for k in ("step", "loss", "val_acc")}, flush=True)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
    secs = time.perf_counter() - t0
    log_chunk("mezo", guard, mem, {"steps_done": step, "train_seconds": round(secs, 1),
                                   "replay_seconds": round(replay, 1), "replayed_steps": len(steps)})
    print(json.dumps({"remaining": MEZO["steps"] - step}))


# ---------------------------------------------------------------- summary
def summary() -> dict:
    b = common.read_results("base", OUT)
    lo = read_jsonl(OUT / "lora_log.jsonl")
    me = read_jsonl(OUT / "mezo_steps.jsonl")
    ch = read_jsonl(OUT / "chunks.jsonl")

    def arm(rows, name, cfg):
        cs = [c for c in ch if c["arm"] == name]
        evals = [{"step": 0, "val_acc": b["val_acc"]}] + [{"step": r["step"], "val_acc": r["val_acc"]}
                                                          for r in rows if "val_acc" in r]
        train_s = sum(c["train_seconds"] for c in cs)
        return {"config": cfg, "steps": len(rows), "evals": evals, "final_val_acc": evals[-1]["val_acc"],
                "best_val_acc": max(e["val_acc"] for e in evals),
                "loss_first_50": round(sum(r["loss"] for r in rows[:50]) / 50, 4),
                "loss_last_50": round(sum(r["loss"] for r in rows[-50:]) / 50, 4),
                "seconds_per_step": round(train_s / len(rows), 3),
                "mps_live_peak_gb": max(c["mps_live_peak_gb"] for c in cs),
                "mps_driver_peak_gb": max(c["mps_driver_peak_gb"] for c in cs),
                "peak_rss_gb": max(c["peak_rss_gb"] for c in cs),
                "chunks": len(cs), "longest_chunk_minutes": max(c["minutes"] for c in cs)}

    out = {"model": MODEL_ID, "revision": MODEL_REVISION, "data": f"{SST2_ID}@{SST2_REVISION[:8]}",
           "prompt": TEMPLATE + " [terrible|great]", "n_train": N_TRAIN, "n_val": b["n_val"], "batch": BATCH,
           "params": b["params"], "base_val_acc": b["val_acc"], "base_memory": b["memory"],
           "lora": arm(lo, "lora", LORA), "mezo": arm(me, "mezo", MEZO),
           "mezo_trajectory_bytes": (OUT / "mezo_steps.jsonl").stat().st_size}
    target = out["lora"]["final_val_acc"]
    out["steps_to_lora_final"] = {
        k: next((e["step"] for e in out[k]["evals"] if e["val_acc"] >= target), None) for k in ("lora", "mezo")}
    # The driver peak is the fair comparison: live tensors are polled between operations, after a
    # no-grad forward pass has freed its activations (MeZO) but while backprop's are still held (LoRA).
    out["mezo_memory_share_of_lora"] = {
        "driver": round(out["mezo"]["mps_driver_peak_gb"] / out["lora"]["mps_driver_peak_gb"], 3),
        "live_polled": round(out["mezo"]["mps_live_peak_gb"] / out["lora"]["mps_live_peak_gb"], 3)}
    out["replay_seconds_last_chunk"] = [c for c in ch if c["arm"] == "mezo"][-1]["replay_seconds"]
    common.write_results("library", out, OUT)
    print(json.dumps({k: out[k] for k in ("base_val_acc", "steps_to_lora_final", "mezo_memory_share_of_lora")}))
    return out


if __name__ == "__main__":
    {"base": base, "lora": lora, "mezo": mezo, "summary": summary}[sys.argv[1]]()
