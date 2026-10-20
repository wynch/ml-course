"""Section 5: the library way. PEFT's LoRA on SmolLM2-360M-Instruct learns
GSM8K's worked solutions; a fixed 500-item probe measures what it forgets;
forward KL to the start model is measured on the new task, as in section 3.
Then where the memory of one training step goes.

    uv run python src/library.py memory     # one step, measured; with and without checkpointing
    uv run python src/library.py train      # repeat until it prints "remaining: 0"

Each `train` call stops after about four minutes and picks up where the last
stopped; adapters and optimiser state wait in `outputs/` (git-ignored). The
model is module 17's, at its pinned revision, in bfloat16 on the GPU (MPS).
"""

from __future__ import annotations

import argparse
import json
import time

import common  # noqa: F401  (laptop guard before torch)
import numpy as np
import torch
import torch.nn.functional as F

from common import GSM8K, MODULE, RESULTS, SMOLTALK, SQUAD, Guard, parquet, write_results

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"     # module 17's pin
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
OUT = MODULE / "python" / "outputs" / "library"
RUNS = RESULTS / "library"
CHUNK_SECONDS = 240

N_NEW_EVAL = 200          # GSM8K test items scored for the new task
N_PROBE_HALF = 250        # 250 SQuAD answers + 250 chat replies = the 500-item probe
MAX_LEN = 512
BATCH = 4                 # sequences per step, as two micro-batches of 2
MICRO = 2                 # batch 4 at once peaked at 4.44 GB on the GPU, over the 4 GB guard
STEPS = 300
EVAL_AT = (0, 50, 100, 200, 300)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ARMS = {
    "lora8_gentle": dict(r=8, lr=1e-4),
    "lora8": dict(r=8, lr=1e-3),
    "lora64": dict(r=64, lr=1e-3),
    "lora8_replay": dict(r=8, lr=1e-3, replay=1),     # 1 of the 4 sequences per step is a chat
}


# ------------------------------------------------------------------ data

def chat_ids(tok, messages: list[dict]) -> tuple[list[int], int]:
    """Token ids of a chat, and where the last assistant reply starts."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    assert ids[:len(p)] == p, "chat template is not a prefix"
    if len(ids) > MAX_LEN:                       # cut from the left, keep the reply
        cut = len(ids) - MAX_LEN
        ids, start = ids[cut:], max(1, len(p) - cut)
    else:
        start = len(p)
    return ids, start


def gsm_messages(r: dict) -> list[dict]:
    return [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}]


def build_sets(tok) -> dict[str, list[tuple[list[int], int]]]:
    rng = np.random.default_rng(29)
    gsm_test = parquet(GSM8K, "main/test-00000-of-00001.parquet")
    new = [chat_ids(tok, gsm_messages(gsm_test[i])) for i in rng.choice(len(gsm_test), N_NEW_EVAL, replace=False)]
    squad = parquet(SQUAD, "plain_text/validation-00000-of-00001.parquet")
    pick = rng.choice(len(squad), N_PROBE_HALF, replace=False)
    qa = [chat_ids(tok, [
        {"role": "user", "content": f"Answer with a few words from the text.\n\nText: {squad[i]['context']}\n\n"
                                    f"Question: {squad[i]['question']}"},
        {"role": "assistant", "content": squad[i]["answers"]["text"][0]}]) for i in pick]
    talk = parquet(SMOLTALK, "data/everyday-conversations/test-00000-of-00001.parquet")
    turns = [(c, j) for c, row in enumerate(talk) for j, m in enumerate(row["messages"]) if m["role"] == "assistant"]
    pick = rng.choice(len(turns), N_PROBE_HALF, replace=False)
    chat = [chat_ids(tok, talk[turns[k][0]]["messages"][:turns[k][1] + 1]) for k in pick]
    return {"new": new, "probe_qa": qa, "probe_chat": chat}


def train_stream(tok, seed: int):
    """Endless shuffled GSM8K training chats; and smoltalk training chats for replay."""
    rng = np.random.default_rng(seed)
    gsm = parquet(GSM8K, "main/train-00000-of-00001.parquet")
    talk = parquet(SMOLTALK, "data/everyday-conversations/train-00000-of-00001.parquet")

    def gen(rows, make):
        while True:
            for i in rng.permutation(len(rows)):
                yield chat_ids(tok, make(rows[i]))
    def upto_last_reply(r: dict) -> list[dict]:
        """A smoltalk chat cut after its last assistant turn; a few end on a user turn."""
        m = r["messages"]
        last = max(i for i, x in enumerate(m) if x["role"] == "assistant")
        return m[:last + 1]
    return gen(gsm, gsm_messages), gen(talk, upto_last_reply)


def collate(items, pad_id: int):
    n = max(len(ids) for ids, _ in items)
    x = torch.full((len(items), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), n), dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.bool)       # positions whose next token is scored
    for i, (ids, start) in enumerate(items):
        x[i, :len(ids)] = torch.tensor(ids)
        att[i, :len(ids)] = 1
        mask[i, start - 1:len(ids) - 1] = True
    return x, att, mask


# ------------------------------------------------------------------ model

def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    # Loaded on the CPU (memory-mapped, so few pages become resident), then moved. Loading with
    # device_map="mps" instead raised peak RSS from 0.67 to 1.35 GB in the memory measurement.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=DTYPE)
    return model.to(DEVICE), tok


def with_lora(model, r: int):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def free() -> None:
    """Hand cached GPU blocks back to the driver. Chats differ in length, so without this
    the allocator keeps one block per shape and the driver's share climbs past the guard
    (results/library_first_try.json)."""
    if DEVICE == "mps":
        torch.mps.empty_cache()


def loss_on(model, x, att, mask) -> torch.Tensor:
    m = mask[:, :-1]
    logits = model(input_ids=x, attention_mask=att).logits[:, :-1][m].float()   # reply positions only
    return F.cross_entropy(logits, x[:, 1:][m])


@torch.no_grad()
def measure(model, tok, sets) -> dict:
    """Mean NLL per reply token on each set, and forward KL(start || model) per reply token.

    The start model is the same network with the adapter switched off.
    """
    model.eval()
    out = {}
    for name, items in sets.items():
        nll, kl, n = 0.0, 0.0, 0
        for i in range(0, len(items), MICRO):
            x, att, mask = collate(items[i:i + MICRO], tok.pad_token_id)
            x, att, mask = x.to(DEVICE), att.to(DEVICE), mask.to(DEVICE)
            m = mask[:, :-1]
            y = x[:, 1:][m]
            lp = F.log_softmax(model(input_ids=x, attention_mask=att).logits[:, :-1][m].float(), -1)
            with model.disable_adapter():
                lp0 = F.log_softmax(model(input_ids=x, attention_mask=att).logits[:, :-1][m].float(), -1)
            nll += float(-lp.gather(-1, y[:, None]).sum())
            kl += float((lp0.exp() * (lp0 - lp)).sum())
            n += int(m.sum())
            del lp, lp0
            free()
        out[f"{name}_nll"] = round(nll / n, 5)
        out[f"kl_{name}"] = round(kl / n, 5)
        out[f"{name}_tokens"] = n
    out["probe_nll"] = round((out["probe_qa_nll"] + out["probe_chat_nll"]) / 2, 5)
    out["kl_probe"] = round((out["kl_probe_qa"] + out["kl_probe_chat"]) / 2, 5)
    return out


# ------------------------------------------------------------------ training

def train(arm: str, budget: float, g: Guard) -> bool:
    """Advance one arm by up to `budget` seconds; True when it has finished."""
    spec = ARMS[arm]
    folder = OUT / arm
    state_file = folder / "state.json"
    model, tok = load()
    model = with_lora(model, spec["r"])
    # Saved activations grow with chat length; a first try without checkpointing ran out of
    # memory at the 5.92 GB cap on GSM8K's longer chats. Checkpointing keeps them small.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=spec["lr"], weight_decay=0.0)
    state = {"step": 0, "log": [], "seconds": 0.0}
    if state_file.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = json.loads(state_file.read_text())
        set_peft_model_state_dict(model, load_file(folder / "adapter.safetensors"))
        opt.load_state_dict(torch.load(folder / "opt.pt", weights_only=False))
    sets = build_sets(tok)
    new_stream, chat_stream = train_stream(tok, seed=29)
    for _ in range(state["step"] * (BATCH - spec.get("replay", 0))):       # skip what was seen
        next(new_stream)
    for _ in range(state["step"] * spec.get("replay", 0)):
        next(chat_stream)
    t0 = time.perf_counter()
    while True:
        if state["step"] in EVAL_AT and all(r["step"] != state["step"] for r in state["log"]):
            m = measure(model, tok, sets)
            state["log"].append({"step": state["step"]} | m)
            print(f"  {arm} step {state['step']}: new {m['new_nll']:.3f}, probe {m['probe_nll']:.3f}, "
                  f"KL new {m['kl_new']:.3f}")
            g.tick()
        if state["step"] >= STEPS or time.perf_counter() - t0 > budget:
            break
        model.train()
        items = [next(new_stream) for _ in range(BATCH - spec.get("replay", 0))]
        items += [next(chat_stream) for _ in range(spec.get("replay", 0))]
        opt.zero_grad(set_to_none=True)
        for j in range(0, BATCH, MICRO):
            x, att, mask = (t.to(DEVICE) for t in collate(items[j:j + MICRO], tok.pad_token_id))
            (loss_on(model, x, att, mask) * MICRO / BATCH).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        state["step"] += 1
        g.tick()
        free()
    state["seconds"] += time.perf_counter() - t0
    state["peak_mps_gb"] = max(state.get("peak_mps_gb", 0), round(g.mps_peak / 2**30, 2))
    state["peak_rss_gb"] = max(state.get("peak_rss_gb", 0), round(g.rss_peak_bytes() / 2**30, 2))
    if state["step"] >= STEPS and any(r["step"] == STEPS for r in state["log"]):
        first, last = state["log"][0], state["log"][-1]
        write_results(arm, {
            "model": MODEL_ID, "revision": MODEL_REVISION, "arm": arm, "spec": spec | {"alpha": 2 * spec["r"]},
            "targets": TARGETS, "steps": STEPS, "batch": BATCH, "micro_batch": MICRO, "max_len": MAX_LEN, "dtype": "bfloat16", "gradient_checkpointing": True,
            "trainable_parameters": sum(p.numel() for p in params),
            "sets": {"new": f"{N_NEW_EVAL} GSM8K test solutions",
                     "probe_qa": f"{N_PROBE_HALF} SQuAD validation answers",
                     "probe_chat": f"{N_PROBE_HALF} smoltalk everyday-conversations test replies"},
            "log": state["log"],
            "final": {"new_gain": round(first["new_nll"] - last["new_nll"], 5),
                      "forgetting": round(last["probe_nll"] - first["probe_nll"], 5),
                      "kl_new": last["kl_new"], "kl_probe": last["kl_probe"]},
            "seconds": round(state["seconds"], 1),
            "peak_mps_driver_gb": state["peak_mps_gb"], "peak_rss_gb": state["peak_rss_gb"],
        }, RUNS)
        return True
    folder.mkdir(parents=True, exist_ok=True)
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    save_file(get_peft_model_state_dict(model), folder / "adapter.safetensors")
    torch.save(opt.state_dict(), folder / "opt.pt")
    g.tick()                                   # saving copies the adapter and Adam state to the CPU
    state["peak_mps_gb"] = max(state["peak_mps_gb"], round(g.mps_peak / 2**30, 2))
    state["peak_rss_gb"] = max(state["peak_rss_gb"], round(g.rss_peak_bytes() / 2**30, 2))
    state_file.write_text(json.dumps(state))
    print(f"  {arm}: step {state['step']} of {STEPS}")
    return False


# ------------------------------------------------------------------ memory

def nbytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def memory() -> None:
    """Where the memory of one LoRA training step goes, on the GPU, with and without checkpointing."""
    g = Guard("library.py memory", DEVICE)
    model, tok = load()
    n_params = sum(p.numel() for p in model.parameters())
    gsm = parquet(GSM8K, "main/train-00000-of-00001.parquet")
    items = [chat_ids(tok, gsm_messages(gsm[i])) for i in range(MICRO)]
    x, att, mask = (t.to(DEVICE) for t in collate(items, tok.pad_token_id))
    out = {"model": MODEL_ID, "revision": MODEL_REVISION, "device": DEVICE, "batch": MICRO,
           "tokens_per_sequence": x.shape[1], "base_parameters": n_params,
           "base_weight_bytes": nbytes(model.parameters()), "runs": {}}
    model = with_lora(model, 8)
    params = [p for p in model.parameters() if p.requires_grad]
    out["lora_r8_parameters"] = sum(p.numel() for p in params)
    out["lora_r8_weight_bytes"] = nbytes(params)
    for ckpt in (False, True):
        if ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        opt = torch.optim.AdamW(params, lr=1e-4)
        model.train()
        for rep in range(3):                     # the third step is the one measured
            torch.mps.synchronize(); torch.mps.empty_cache()
            before = torch.mps.current_allocated_memory()
            t = time.perf_counter()
            loss = loss_on(model, x, att, mask)
            torch.mps.synchronize()
            after_fwd = torch.mps.current_allocated_memory()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.mps.synchronize()
            opt.step()
            torch.mps.synchronize()
            secs = time.perf_counter() - t
            g.tick()
            del loss
        state = [v for s in opt.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0]
        out["runs"]["checkpointing" if ckpt else "plain"] = {
            "saved_for_backward_bytes": after_fwd - before,
            "lora_grad_bytes": nbytes([p.grad for p in params if p.grad is not None]),
            "adam_state_bytes": nbytes(state),
            "step_seconds": round(secs, 3),
        }
        del opt
    p = n_params
    out["full_finetune_on_paper"] = {
        "note": "not run: over the module's 4 GB guard",
        "float32_weights_grads_adam_bytes": 16 * p,
        "bfloat16_weights_grads_float32_adam_bytes": (2 + 2 + 8) * p,
    }
    out["guard"] = g.report()
    write_results("library_memory", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["memory", "train"])
    args = ap.parse_args()
    if args.stage == "memory":
        memory()
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [a for a in ARMS if not (RUNS / f"{a}.json").exists()]
    g = Guard("library.py train", DEVICE)
    t0 = time.perf_counter()
    while todo:
        left = CHUNK_SECONDS - (time.perf_counter() - t0)
        if left < 60:
            break
        if train(todo[0], left, g):
            todo.pop(0)
        else:
            break
    print(f"remaining: {len(todo)}")


if __name__ == "__main__":
    main()
