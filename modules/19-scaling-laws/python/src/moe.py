"""Section 5: mixture of experts. Module 04's GPT with its MLP swapped for experts.

    uv run python src/moe.py              # every arm, two seeds each; writes results/moe.json
    uv run python src/moe.py --arms moe8_all   # one arm, merged into results/moe.json
    uv run python src/moe.py --figures    # redraw figures/moe.png from results/
    uv run python src/moe.py --controls   # review checks (seed 2, dense_gated, rates); results/moe_controls.json

Module 04's model code is imported from module 04 itself, not copied. Width
128, 4 layers, 4 heads, context 128: the shape of section 1's family, so the
dense model lands on section 1's curve. The recipe is section 1's: 977 steps
of 32 x 128 characters of wikitext, none seen twice, learning rate 3e-4
after 100 warm-up steps, no decay, dropout 0.

Seven arms differ only in the feed-forward part of each block:

    dense        module 04's MLP, 128 -> 512 -> 128
    dense_wide   one MLP 128 -> 2048 -> 128: the MoE arms' total size
    moe8         8 experts of 128 -> 256 -> 128, top-2: dense's active size, dense_wide's total
    moe8_bal     the same, plus the Switch balancing loss, weight 0.01
    moe32_bal    32 experts of width 64, top-8: fine-grained, same active and total
    moe30s_bal   1 shared expert of width 128 and 30 routed of width 64, top-6: same again
    moe8_all     8 experts of width 256, all 8 run, mixed by the router's weights: no sparsity
                 (a control, added after the first six ran: is the gain the routing or the sparsity?)
    dense_gated  8 experts of width 64, all 8 run, mixed by the router: dense's size with the gate
                 (a review control, --controls only)

The router is a linear map to one score per expert, softmax, keep the top k,
renormalise their weights to sum to 1 (Mixtral's rule). For speed on the
Apple GPU, training runs every expert on every token and multiplies the
unchosen ones by zero; ``MoE.sparse = True`` runs each expert on its own
tokens only, and gives the same output (the tests check it). So wall time
here is not the saving a real MoE kernel gets; the parameter counts are.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (BATCH, BLOCK, LR, MODULE, N_HEAD, N_LAYER, STEPS, WARMUP, WEIGHT_DECAY, Guard, get_device,
                    read_results, tokens_at, write_results)

M04 = MODULE.parent / "04-attention-transformer"
D_MODEL = 128
SEEDS = (0, 1)
EVAL_STEPS = (61, 122, 244, 488, 977)     # 0.25, 0.5, 1, 2 and 4M tokens

ARMS = {
    "dense":      dict(kind="dense", hidden=512),
    "dense_wide": dict(kind="dense", hidden=2048),
    "moe8":       dict(kind="moe", experts=8, k=2, hidden=256, balance=0.0),
    "moe8_bal":   dict(kind="moe", experts=8, k=2, hidden=256, balance=0.01),
    "moe32_bal":  dict(kind="moe", experts=32, k=8, hidden=64, balance=0.01),
    "moe30s_bal": dict(kind="moe", experts=30, k=6, hidden=64, balance=0.01, shared_hidden=128),
    "moe8_all":   dict(kind="moe", experts=8, k=8, hidden=256, balance=0.0),
    # review control, run by --controls only: dense's size, the router's gate, no sparsity
    "dense_gated": dict(kind="moe", experts=8, k=8, hidden=64, balance=0.0),
}
MAIN_ARMS = ("dense", "dense_wide", "moe8", "moe8_bal", "moe32_bal", "moe30s_bal", "moe8_all")

# --controls: checks added in review, after every number above was in. Written to results/moe_controls.json.
CONTROL_SEED = 2                                   # a third seed for every arm, at the recipe's rate
HIGH_LR = 1e-3                                     # every arm again at this rate, seeds 0 and 1
SWEEP = {"dense": (1e-4, 6e-4, 2e-3), "dense_wide": (1e-4, 6e-4, 2e-3), "moe8_bal": (1e-4, 6e-4, 2e-3)}


def m04():
    """Module 04's model.py, loaded by path under its own name so it never shadows another module."""
    if "m04_model" in sys.modules:
        return sys.modules["m04_model"]
    spec = importlib.util.spec_from_file_location("m04_model", M04 / "python" / "src" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m04_model"] = mod      # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the parts
class Experts(nn.Module):
    """E feed-forward nets of module 04's shape, d -> h -> d with GELU, stored as stacked weights."""

    def __init__(self, n: int, d: int, h: int, n_layer: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n, d, h) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(n, h))
        # module 04's init for every residual projection: std 0.02 / sqrt(2 * layers)
        self.w2 = nn.Parameter(torch.randn(n, h, d) * 0.02 / math.sqrt(2 * n_layer))
        self.b2 = nn.Parameter(torch.zeros(n, d))

    def one(self, e: int, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x @ self.w1[e] + self.b1[e]) @ self.w2[e] + self.b2[e]

    def size_one(self) -> int:
        n, d, h = self.w1.shape
        return d * h + h + h * d + d


def route(logits: torch.Tensor, k: int):
    """Softmax over the experts, keep the top k, renormalise. Returns probs (N, E), weights and ids (N, k)."""
    probs = logits.softmax(-1)
    top_w, top_i = probs.topk(k, dim=-1)
    return probs, top_w / top_w.sum(-1, keepdim=True), top_i


def balance_loss(probs: torch.Tensor, top_i: torch.Tensor) -> torch.Tensor:
    """Switch Transformer's term, for top-k: E * sum_e f_e * P_e.

    f_e: share of the N*k routed slots that went to expert e (no gradient).
    P_e: mean router probability of e. It is 1 when both are uniform, and
    larger the more f and P pile onto the same experts.
    """
    E = probs.shape[-1]
    f = torch.bincount(top_i.flatten(), minlength=E).to(probs.dtype) / top_i.numel()
    return E * (f * probs.mean(0)).sum()


class MoE(nn.Module):
    """A router, E routed experts, top-k per token, an optional shared expert every token runs."""

    def __init__(self, d: int, experts: int, k: int, hidden: int, n_layer: int, shared_hidden: int = 0):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = Experts(experts, d, hidden, n_layer)
        self.shared = Experts(1, d, shared_hidden, n_layer) if shared_hidden else None
        self.aux = torch.zeros(())
        self.last_ids: torch.Tensor | None = None

    sparse = False      # True: run each expert on its own tokens only (slow on MPS; the same numbers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        probs, w, ids = route(self.router(flat), self.k)
        self.aux = balance_loss(probs, ids)
        self.last_ids = ids.detach().reshape(B, T, self.k)
        if self.sparse:
            out = self._dispatch(flat, w, ids)
        else:
            # Every expert on every token, then a gate that is zero for the experts not chosen.
            # Same output and gradients as the dispatch; costs what dense_wide costs.
            gate = torch.zeros_like(probs).scatter(1, ids, w)                       # (N, E)
            ex = self.experts
            h = F.gelu(torch.einsum("nd,edh->neh", flat, ex.w1) + ex.b1)
            out = torch.einsum("ne,ned->nd", gate, torch.einsum("neh,ehd->ned", h, ex.w2) + ex.b2)
        if self.shared is not None:
            out = out + self.shared.one(0, flat)
        return out.reshape(B, T, C)

    def _dispatch(self, flat, w, ids):
        # Sort the N*k (token, slot) pairs by expert, then run each expert on its slice.
        e_of = ids.flatten()
        order = e_of.argsort()
        tok = order // self.k
        counts = torch.bincount(e_of, minlength=self.experts.w1.shape[0]).tolist()   # one host sync
        out = torch.zeros_like(flat)
        start = 0
        for e, n in enumerate(counts):
            if n:
                t = tok[start:start + n]
                y = self.experts.one(e, flat[t]) * w.flatten()[order[start:start + n], None]
                out.index_add_(0, t, y)
            start += n
        return out


class DenseMLP(nn.Module):
    """Module 04's MLP at another width."""

    def __init__(self, d: int, hidden: int, n_layer: int):
        super().__init__()
        self.ff = Experts(1, d, hidden, n_layer)

    def forward(self, x):
        return self.ff.one(0, x)


# ------------------------------------------------------------------ the model
def build(arm: str, vocab: int, seed: int):
    torch.manual_seed(seed)
    mod = m04()
    cfg = mod.GPTConfig(vocab_size=vocab, block_size=BLOCK, n_layer=N_LAYER, n_head=N_HEAD, d_model=D_MODEL,
                        dropout=0.0)
    model = mod.GPT(cfg)
    a = ARMS[arm]
    if not (a["kind"] == "dense" and a["hidden"] == 4 * D_MODEL):
        for blk in model.blocks:
            if a["kind"] == "dense":
                blk.mlp = DenseMLP(D_MODEL, a["hidden"], N_LAYER)
            else:
                blk.mlp = MoE(D_MODEL, a["experts"], a["k"], a["hidden"], N_LAYER, a.get("shared_hidden", 0))
    return model


def counts(model, arm: str) -> dict:
    """Parameters: all, non-embedding (section 1's N), and the non-embedding share one token uses."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.wte.weight.numel() + model.wpe.weight.numel()
    n = total - emb
    a = ARMS[arm]
    idle = 0
    if a["kind"] == "moe":
        idle = N_LAYER * (a["experts"] - a["k"]) * model.blocks[0].mlp.experts.size_one()
    return {"total": total, "n_non_embedding": n, "n_active": n - idle}


def aux_loss(model) -> torch.Tensor:
    return sum(b.mlp.aux for b in model.blocks if isinstance(b.mlp, MoE))


# ------------------------------------------------------------------ data and loss, as section 1
def held_batches(held: np.ndarray):
    n = (len(held) - 1) // BLOCK
    return (torch.from_numpy(held[: n * BLOCK].reshape(n, BLOCK)),
            torch.from_numpy(held[1: n * BLOCK + 1].reshape(n, BLOCK)))


@torch.no_grad()
def held_loss(model, hx, hy, chunk: int = 130) -> float:
    was = model.training
    model.eval()
    tot = 0.0
    for i in range(0, len(hx), chunk):
        _, loss = model(hx[i:i + chunk], hy[i:i + chunk])
        tot += loss.item() * len(hx[i:i + chunk])
    model.train(was)
    return tot / len(hx)


def train_one(arm: str, seed: int, train, held, vocab: int, dev, guard: Guard, lr: float = LR):
    model = build(arm, vocab, seed).to(dev)
    weight = ARMS[arm].get("balance", 0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=WEIGHT_DECAY)
    w = BLOCK + 1
    n_win = len(train) // w
    order = np.random.default_rng(seed).permutation(n_win)
    wins = torch.from_numpy(train[: n_win * w].reshape(n_win, w))
    hx, hy = (t.to(dev) for t in held_batches(held))
    curve, aux_log = [], []
    model.train()
    t_train = 0.0
    for step in range(1, STEPS + 1):
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / WARMUP)
        b = wins[order[(step - 1) * BATCH: step * BATCH]].to(dev)
        _, loss = model(b[:, :-1], b[:, 1:].contiguous())
        aux = aux_loss(model) if ARMS[arm]["kind"] == "moe" else torch.zeros((), device=dev)
        opt.zero_grad(set_to_none=True)
        (loss + weight * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lv = loss.item()                  # waits for the GPU, so the clock is honest
        t_train += time.perf_counter() - t0
        if step % 50 == 0 or step == 1:
            aux_log.append([step, round(float(aux.detach()) / N_LAYER, 5)])
        if not math.isfinite(lv):
            raise FloatingPointError(f"{arm} seed {seed} lr {lr}: loss {lv} at step {step}")
        if step in EVAL_STEPS:
            curve.append({"step": step, "tokens": tokens_at(step), "held_loss": round(held_loss(model, hx, hy), 6)})
            guard.tick()
    c = counts(model, arm)
    print(f"  {arm:11s} seed {seed}: N {c['n_non_embedding']:,} active {c['n_active']:,}  "
          f"held-out {curve[-1]['held_loss']:.4f}  {1000 * t_train / STEPS:.0f} ms/step")
    return {"seed": seed, "params": c, "curve": curve, "train_loss_last": round(lv, 5),
            "balance_loss_per_layer": aux_log, "ms_per_step": round(1000 * t_train / STEPS, 1)}, model


# ------------------------------------------------------------------ where tokens go
CLASSES = ("space", "newline", "punctuation", "digit", "capital", "vowel", "consonant", "other")


def char_class(c: str) -> str:
    if c == " ":
        return "space"
    if c == "\n":
        return "newline"
    if c.isdigit():
        return "digit"
    if c.isalpha() and c.isascii():
        if c.isupper():
            return "capital"
        return "vowel" if c in "aeiou" else "consonant"
    if c in ".,;:!?'\"()-@=<>[]/&":
        return "punctuation"
    return "other"


@torch.no_grad()
def routing(model, hx, vocab, k: int) -> dict:
    """Over every held-out token: each expert's share of the routed slots, per layer, and what it reads.

    For each expert: the share of its slots by class of the input character
    against that class's share of all tokens (the lift), the three characters
    it takes most often, and the mean position of its tokens in their word
    (0 = first letter).
    """
    model.eval()
    ids_by_layer = [[] for _ in model.blocks]
    for i in range(0, len(hx), 130):
        model(hx[i:i + 130])
        for li, b in enumerate(model.blocks):
            ids_by_layer[li].append(b.mlp.last_ids.cpu())
    x = hx.cpu().flatten().numpy()
    chars = np.array([vocab.chars[j] if j < vocab.unk else "?" for j in range(vocab.size)])
    cls = np.array([CLASSES.index(char_class(c)) for c in chars])[x]
    # position in word: characters since the last space or newline
    pos = np.zeros_like(x)
    sep = np.isin(chars[x], [" ", "\n"])
    for t in range(1, len(x)):
        pos[t] = 0 if sep[t - 1] else pos[t - 1] + 1
    base = np.bincount(cls, minlength=len(CLASSES)) / len(cls)
    layers = []
    for li, parts in enumerate(ids_by_layer):
        ids = torch.cat(parts).reshape(-1, k).numpy()
        E = model.blocks[li].mlp.experts.w1.shape[0]
        share = np.bincount(ids.flatten(), minlength=E) / ids.size
        experts = []
        for e in range(E):
            tok = (ids == e).any(1)
            if not tok.any():
                experts.append({"share": 0.0})
                continue
            cs = np.bincount(cls[tok], minlength=len(CLASSES)) / tok.sum()
            lift = {CLASSES[c]: round(float(cs[c] / base[c]), 2) for c in range(len(CLASSES)) if base[c] > 0}
            top = np.bincount(x[tok], minlength=vocab.size).argsort()[::-1][:3]
            experts.append({"share": round(float(share[e]), 4),
                            "class_share": {CLASSES[c]: round(float(cs[c]), 3) for c in range(len(CLASSES))},
                            "lift": lift, "top_chars": [chars[j] for j in top],
                            "mean_pos_in_word": round(float(pos[tok].mean()), 2)})
        fair = 1 / E
        layers.append({"experts": experts, "max_over_fair": round(float(share.max() / fair), 2),
                       "min_over_fair": round(float(share.min() / fair), 2),
                       "unused": int((share == 0).sum()),
                       "load_entropy_over_max": round(float(-(share[share > 0] * np.log(share[share > 0])).sum()
                                                            / math.log(E)), 3)})
    return {"base_class_share": {CLASSES[c]: round(float(base[c]), 4) for c in range(len(CLASSES))},
            "layers": layers}


def run(only: list[str] | None = None) -> None:
    """Train every arm, or with ``only`` just those arms, merged into the existing results/moe.json."""
    from corpus import splits

    guard = Guard("moe.py" + (" --arms " + " ".join(only) if only else ""))
    dev = get_device()
    train, held, vocab = splits()
    guard.tick("load data")
    hx = held_batches(held)[0].to(dev)
    arms = {}
    for arm in (only or MAIN_ARMS):
        runs = []
        for seed in SEEDS:
            r, model = train_one(arm, seed, train, held, vocab.size, dev, guard)
            if ARMS[arm]["kind"] == "moe":
                r["routing"] = routing(model, hx, vocab, ARMS[arm]["k"])
            runs.append(r)
            del model
            guard.tick(f"{arm} seed {seed}")
        arms[arm] = {"spec": ARMS[arm], "runs": runs}
    if only:
        out = read_results("moe")
        out["arms"].update(arms)
        out.setdefault("guard_extra", []).append(guard.report())
    else:
        out = {
            "d_model": D_MODEL, "n_layer": N_LAYER, "n_head": N_HEAD, "block": BLOCK, "batch": BATCH,
            "steps": STEPS, "lr": LR, "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "vocab": vocab.size,
            "device": dev.type,
            "model_code": "modules/04-attention-transformer/python/src/model.py, imported by path",
            "arms": arms, "guard": guard.report(),
        }
    write_results("moe", out)


def controls() -> None:
    """Checks added in review: a third seed, a dense model with the router's gate, and the learning rate.

    Every run is train_one, unchanged but for the rate. Writes results/moe_controls.json.
    """
    from corpus import splits

    guard = Guard("moe.py --controls")
    dev = get_device()
    train, held, vocab = splits()
    plan = [(a, CONTROL_SEED, LR) for a in MAIN_ARMS] + [("dense_gated", s, LR) for s in (0, 1, 2)]
    plan += [(a, s, HIGH_LR) for a in MAIN_ARMS + ("dense_gated",) for s in (0, 1)]
    plan += [(a, 0, lr) for a, lrs in SWEEP.items() for lr in lrs]
    runs = []
    for arm, seed, lr in plan:
        try:
            r, model = train_one(arm, seed, train, held, vocab.size, dev, guard, lr=lr)
            del model
            runs.append({"arm": arm, "seed": seed, "lr": lr, "held_loss": r["curve"][-1]["held_loss"],
                         "params": r["params"], "curve": r["curve"], "train_loss_last": r["train_loss_last"]})
        except FloatingPointError as e:
            print("  ", e)
            runs.append({"arm": arm, "seed": seed, "lr": lr, "held_loss": None, "diverged": str(e)})
        guard.tick(f"{arm} seed {seed} lr {lr}")
    write_results("moe_controls", {"steps": STEPS, "recipe_lr": LR, "high_lr": HIGH_LR, "runs": runs,
                                   "dense_gated": ARMS["dense_gated"], "guard": guard.report()})


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import FIGURES, SIZES

    r = read_results("moe")
    fit = read_results("fit")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6))
    # left: held-out loss against N, section 1's six sizes in grey
    ns, ls = [], []
    for d in SIZES:
        t = read_results(f"train_d{d}")["runs"][0]
        ns.append(t["n_non_embedding"])
        ls.append(t["curve"][-1]["held_loss"])
    ax.plot(ns, ls, "o-", color="0.7", label="section 1: dense, 4 layers, widths 32-256")
    a, log_c = fit["l_of_n_main"]["exponent"], fit["l_of_n_main"]["a"]      # log L = a - alpha log N
    xs = np.geomspace(min(ns), max(ns), 50)
    ax.plot(xs, np.exp(log_c) * xs ** (-a), ":", color="0.6", label=f"section 2's fit, alpha {a:.3f}")
    style = {"dense": ("C0", "s"), "dense_wide": ("C1", "s"), "moe8": ("C3", "o"), "moe8_bal": ("C2", "o"),
             "moe32_bal": ("C4", "D"), "moe30s_bal": ("C5", "^"), "moe8_all": ("C7", "v")}
    for arm, v in r["arms"].items():
        col, mk = style[arm]
        for s, run in enumerate(v["runs"]):
            p, L = run["params"], run["curve"][-1]["held_loss"]
            lab = arm if s == 0 else None
            if v["spec"]["kind"] == "moe":
                ax.plot([p["n_active"], p["n_non_embedding"]], [L, L], "-", color=col, alpha=0.5)
                ax.plot(p["n_active"], L, mk, color=col, label=lab)
                ax.plot(p["n_non_embedding"], L, mk, mfc="none", color=col)
            else:
                ax.plot(p["n_non_embedding"], L, mk, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters N (MoE: filled = active, hollow = total)")
    ax.set_ylabel("held-out loss at 4M tokens (nats/char)")
    ax.set_title("Where the MoE arms land on section 1's curve")
    ax.legend(fontsize=7.5)
    # right: expert load, layer by layer, seed 0, with and without the balancing term
    for arm, off, col in (("moe8", -0.2, "C3"), ("moe8_bal", 0.2, "C2")):
        lay = r["arms"][arm]["runs"][0]["routing"]["layers"]
        for li, L in enumerate(lay):
            sh = sorted((e["share"] for e in L["experts"]), reverse=True)
            xs = li + off + np.linspace(-0.15, 0.15, len(sh))
            bx.bar(xs, np.array(sh) * len(sh), width=0.035, color=col,
                   label=(arm + (" (balancing term)" if "bal" in arm else " (no term)")) if li == 0 else None)
    bx.axhline(1, color="0.4", ls=":", lw=1)
    bx.set_xticks(range(N_LAYER), [f"layer {i}" for i in range(N_LAYER)])
    bx.set_ylabel("share of routed slots / fair share (1/8)")
    bx.set_title("Expert load, 8 experts per layer, sorted, seed 0")
    bx.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "moe.png", dpi=130)
    print("  wrote figures/moe.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true", help="redraw the figure from results/moe.json")
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), help="train only these arms and merge them in")
    ap.add_argument("--controls", action="store_true",
                    help="review checks: a third seed, a gated dense arm, the learning rate; results/moe_controls.json")
    args = ap.parse_args()
    if args.controls:
        controls()
    elif args.figures:
        figures()
    else:
        run(args.arms)
        figures()


if __name__ == "__main__":
    main()
