"""Section 1: at a fixed file size, train in ternary or round a float model?

Four arms, each at two file budgets (0.5 MB and 1.5 MB):

    ternary  BitLinear trained in (1.6 bits per block-linear weight, one 16-bit scale per matrix)
    4-bit    a float model, rounded after training: affine, groups of 32 (5 bits per weight)
    2-bit    the same rule at 2 bits (3 bits per weight)
    16-bit   a float model stored in bf16, not rounded

The embeddings, norms and biases stay in 16 bits in every arm. Each arm gets
shapes that fill the budget to within 10%, so a 2-bit model is wider or deeper
than a 4-bit one of the same size. To keep the baselines fair (Veldanda, 2026,
arXiv:2609.29397) every arm and budget runs the same protocol:

    A. the middle shape at three learning rates (1e-3, 3e-3, 1e-2), seed 0;
       one more rate beyond the end of the grid if the best sits at an end
    B. the other two shapes at the best rate, seed 0
    C. the best shape and rate again with seeds 1 and 2

Choices use the "tune" half of the held-out text; every reported loss uses the
"report" half, which no choice looked at. Each run is cached as one line of
results/matched_runs.jsonl, so the script can stop and resume: one call trains
runs, and at --max-minutes pauses the current run (saved, resumed next call) and exits. When every run is there, it
writes results/matched.json.

Run me:
    uv run python src/matched.py                      # one chunk of at most 4.5 minutes; repeat until it says "done"
    uv run python src/matched.py --figures            # also figures/matched.png once done
"""

from __future__ import annotations

import argparse
import json
import time

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch

from bitlinear import TERNARY_BITS, absmean, block_linears, ternary, ternary_codes
from chars import splits
from common import RESULTS, Guard, get_device, write_results
from compress import quantize_affine
from train import BATCH, BLOCK, STEPS, held_loss, train

BUDGETS = (0.5e6, 1.5e6)
TOL = 0.10
ARMS = ("ternary", "4-bit", "2-bit", "16-bit")
LR_GRID = (1e-3, 3e-3, 1e-2)
LR_BEYOND = {1e-3: 3e-4, 1e-2: 3e-2}
SEEDS = (0, 1, 2)
MAX_LAYERS = 12
GROUP = 32
RUNS = RESULTS / "matched_runs.jsonl"
EVAL_RESERVE = 25        # seconds kept free at a chunk's end to evaluate a finished run


def arm_bytes(arm: str, d: int, n_layer: int, vocab: int) -> int:
    lin = 12 * d * d * n_layer
    other = (vocab + BLOCK) * d + 13 * d * n_layer + 2 * d     # wte, wpe, per block 2 LayerNorms + 4 biases, ln_f
    if arm == "ternary":
        return round(lin * TERNARY_BITS / 8 + 2 * 4 * n_layer + 2 * other)
    if arm == "16-bit":
        return 2 * (lin + other)
    bits = {"4-bit": 4, "2-bit": 2}[arm]
    return round(lin * bits / 8 + lin // GROUP * 4 + 2 * other)


def shapes(arm: str, budget: float, vocab: int) -> list[tuple[int, int]]:
    """Three (d_model, n_layer) shapes within TOL of the budget: the deepest, the middle one, the shallowest.

    Widths are multiples of 32: heads of 32, and groups of 32 divide every row.
    """
    c = [(d, L) for d in range(64, 1025, 32) for L in range(1, MAX_LAYERS + 1)
         if abs(arm_bytes(arm, d, L, vocab) / budget - 1) <= TOL]
    # For each depth keep the width nearest the budget; then pick three spread-out depths.
    best: dict[int, tuple[int, int]] = {}
    for d, L in c:
        if L not in best or abs(arm_bytes(arm, d, L, vocab) - budget) < abs(arm_bytes(arm, *best[L], vocab) - budget):
            best[L] = (d, L)
    depths = sorted(best)
    assert len(depths) >= 3, (arm, budget, depths)
    pick = [depths[-1], depths[len(depths) // 2], depths[0]]
    return [best[L] for L in pick]


def partial_path(k: str):
    """Where a run paused at a chunk's end waits; outside the repo, removed when the run ends."""
    import os
    from pathlib import Path

    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "mlcourse" / "m28" / "partial" / f"{k}.pt"


def kind_of(arm: str) -> str:
    return "ternary" if arm == "ternary" else "float"


def key(kind, d, L, lr, seed) -> str:
    return f"{kind}-d{d}-L{L}-lr{lr:g}-s{seed}"


def load_runs() -> dict[str, dict]:
    if not RUNS.exists():
        return {}
    rows = [json.loads(line) for line in RUNS.read_text().splitlines() if line.strip()]
    return {r["key"]: r for r in rows if "key" in r}


@torch.no_grad()
def swap_weights(model, fn):
    """Replace every block-linear weight by fn(W); return the originals to restore."""
    saved = {}
    for name, mod in block_linears(model):
        saved[name] = mod.weight.detach().clone()
        mod.weight.copy_(fn(mod.weight))
    return saved


@torch.no_grad()
def restore(model, saved):
    for name, mod in block_linears(model):
        mod.weight.copy_(saved[name])


@torch.no_grad()
def bf16_all(model):
    """Every parameter rounded to bf16 and back: what a 16-bit file holds."""
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters():
        p.copy_(p.to(torch.bfloat16).float())
    return saved


@torch.no_grad()
def restore_all(model, saved):
    for n, p in model.named_parameters():
        p.copy_(saved[n])


@torch.no_grad()
def evaluate(model, kind: str, tune, report, device) -> dict:
    """Losses on both held-out halves for every way this run can be stored."""
    out = {}

    def both(tag):
        out[tag] = {"tune": round(held_loss(model, tune, device), 5), "report": round(held_loss(model, report, device), 5)}

    if kind == "float":
        both("fp32")
        s = bf16_all(model)
        both("16-bit")
        restore_all(model, s)
        for bits in (4, 2):
            s = swap_weights(model, lambda W, b=bits: quantize_affine(W, b, GROUP)[0])
            both(f"{bits}-bit")
            restore(model, s)
        return out
    # Ternary. The training forward pass already uses ternary(W) of the fp32 latents ("fp32 export").
    both("ternary")
    # The bf16 export: cast the latents to bf16, then take codes and scale in bf16.
    n_codes, n_flip, n_tie = 0, 0, 0
    for _, mod in block_linears(model):
        W = mod.weight
        Wb = W.to(torch.bfloat16)
        n_codes += W.numel()
        n_flip += int((ternary_codes(W) != ternary_codes(Wb).float()).sum())
        n_tie += int(((Wb / absmean(Wb)).abs() == 0.5).sum())
    out["export"] = {"codes": n_codes, "flipped_by_bf16_cast": n_flip, "on_threshold_in_bf16": n_tie}
    out["ternary-bf16-export"] = export_loss(model, tune, report, device)
    return out


@torch.no_grad()
def export_loss(model, tune, report, device) -> dict:
    """Loss of the bf16 export: codes and scale from bf16-cast latents, 8-bit inputs as in training."""
    from bitlinear import BitLinear

    saved = {}
    for name, mod in block_linears(model):
        saved[name] = mod.weight.detach().clone()
        mod.weight.copy_(ternary(mod.weight.to(torch.bfloat16)).float())   # codes and scale from bf16 latents
        mod.__class__ = FrozenBit
    out = {"tune": round(held_loss(model, tune, device), 5), "report": round(held_loss(model, report, device), 5)}
    for name, mod in block_linears(model):
        mod.weight.copy_(saved[name])
        mod.__class__ = BitLinear
    return out


class FrozenBit(torch.nn.Linear):
    """A BitLinear whose weights are already ternary times scale: 8-bit inputs, weights used as stored."""

    def forward(self, x):
        from bitlinear import act_quant

        return torch.nn.functional.linear(act_quant(x), self.weight, self.bias)


def plan(vocab: int, runs: dict) -> tuple[list[tuple[str, float, int, int, float, int]], bool]:
    """The runs still needed, in protocol order, and whether the protocol is complete."""
    todo = []
    for budget in BUDGETS:
        for arm in ARMS:
            kind = kind_of(arm)
            sh = shapes(arm, budget, vocab)
            mid = sh[1]

            def score(d, L, lr, seed=0):
                r = runs.get(key(kind, d, L, lr, seed))
                if r is None:
                    return None
                return r["losses"][arm if kind == "float" else "ternary"]["tune"]

            # A: learning rates on the middle shape
            grid = list(LR_GRID)
            missing = [lr for lr in grid if score(*mid, lr) is None]
            if missing:
                todo += [(arm, budget, *mid, lr, 0) for lr in missing]
                continue
            best_lr = min(grid, key=lambda lr: score(*mid, lr))
            if best_lr in LR_BEYOND:
                extra = LR_BEYOND[best_lr]
                if score(*mid, extra) is None:
                    todo.append((arm, budget, *mid, extra, 0))
                    continue
                grid.append(extra)
                best_lr = min(grid, key=lambda lr: score(*mid, lr))
            # B: the other shapes at that rate
            missing = [s for s in (sh[0], sh[2]) if score(*s, best_lr) is None]
            if missing:
                todo += [(arm, budget, *s, best_lr, 0) for s in missing]
                continue
            best_shape = min(sh, key=lambda s: score(*s, best_lr))
            # C: two more seeds
            missing = [seed for seed in SEEDS[1:] if score(*best_shape, best_lr, seed) is None]
            todo += [(arm, budget, *best_shape, best_lr, seed) for seed in missing]
    return todo, not todo


def summarise(vocab: int, runs: dict) -> dict:
    cells = []
    for budget in BUDGETS:
        for arm in ARMS:
            kind = kind_of(arm)
            tag = arm if kind == "float" else "ternary"
            sh = shapes(arm, budget, vocab)
            mine = [r for r in runs.values() if r["kind"] == kind and (r["d"], r["n_layer"]) in sh]
            seed0 = [r for r in mine if r["seed"] == 0]
            best = min(seed0, key=lambda r: r["losses"][tag]["tune"])
            seeds = sorted((r for r in mine if (r["d"], r["n_layer"], r["lr"]) == (best["d"], best["n_layer"], best["lr"])),
                           key=lambda r: r["seed"])
            rep = [r["losses"][tag]["report"] for r in seeds]
            cell = {
                "budget_mb": budget / 1e6, "arm": arm, "shapes": [list(s) for s in sh],
                "best": {"d": best["d"], "n_layer": best["n_layer"], "lr": best["lr"]},
                "bytes": arm_bytes(arm, best["d"], best["n_layer"], vocab),
                "params": best["params"], "linear_weights": 12 * best["d"] ** 2 * best["n_layer"],
                "chars_per_param": round(STEPS * BATCH * BLOCK / best["params"], 2),
                "report_by_seed": rep, "report_mean": round(float(np.mean(rep)), 4),
                "report_sd": round(float(np.std(rep, ddof=1)), 4) if len(rep) > 1 else None,
                "tried": [{"d": r["d"], "n_layer": r["n_layer"], "lr": r["lr"],
                           "bytes": arm_bytes(arm, r["d"], r["n_layer"], vocab),
                           "tune": r["losses"][tag]["tune"], "report": r["losses"][tag]["report"]}
                          for r in sorted(seed0, key=lambda r: (r["n_layer"], r["lr"]))],
            }
            if kind == "float":
                cell["before_rounding_report"] = [r["losses"]["fp32"]["report"] for r in seeds]
            else:
                cell["bf16_export_report"] = [r["losses"]["ternary-bf16-export"]["report"] for r in seeds]
                cell["export"] = [r["losses"]["export"] for r in seeds]
            cells.append(cell)
    return {"cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=4.5, help="pause the current run and exit after this")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    guard = Guard("matched.py")
    device = get_device()
    train_ids, tune, report, vocab = splits()
    runs = load_runs()
    todo, done = plan(vocab.size, runs)
    t0 = time.perf_counter()
    trained = 0
    deadline = t0 + args.max_minutes * 60 - EVAL_RESERVE
    while todo and time.perf_counter() < deadline:
        arm, budget, d, L, lr, seed = todo[0]
        kind = kind_of(arm)
        model, info = train(kind, d, L, lr, seed, train_ids, vocab.size, device,
                            ckpt=partial_path(key(kind, d, L, lr, seed)), deadline=deadline)
        if model is None:
            print(f"  {key(kind, d, L, lr, seed)}: paused at the chunk's end, saved to resume")
            break
        losses = evaluate(model, kind, tune, report, device)
        row = {"key": key(kind, d, L, lr, seed), **info, "losses": losses, "for": [arm, budget]}
        with RUNS.open("a") as f:
            f.write(json.dumps(row) + "\n")
        runs[row["key"]] = row
        trained += 1
        tag = arm if kind == "float" else "ternary"
        print(f"  {row['key']:32s} {info['seconds']:6.1f} s  tune {losses[tag]['tune']:.4f}  ({arm}, {budget/1e6} MB)")
        guard.tick(row["key"])           # sample MPS memory while the model is still there
        del model
        common.free()
        todo, done = plan(vocab.size, runs)
    g = guard.report()
    with RUNS.open("a") as f:
        f.write(json.dumps({"chunk": g, "runs_trained": trained}) + "\n")
    if not done:
        print(f"not done: {len(todo)} runs queued next (more may follow); call again")
        return
    out = {"setup": {"budgets_mb": [b / 1e6 for b in BUDGETS], "tolerance": TOL, "arms": list(ARMS),
                     "lr_grid": list(LR_GRID), "seeds": list(SEEDS), "steps": STEPS, "batch": BATCH, "block": BLOCK,
                     "chars_seen": STEPS * BATCH * BLOCK, "group": GROUP, "ternary_bits": TERNARY_BITS,
                     "vocab": vocab.size, "tune_chars": len(tune), "report_chars": len(report),
                     "device": str(device)},
           **summarise(vocab.size, runs)}
    chunks = [json.loads(line)["chunk"] for line in RUNS.read_text().splitlines() if '"chunk"' in line]
    out["chunks"] = {"count": len(chunks), "longest_minutes": max(c["minutes"] for c in chunks),
                     "total_minutes": round(sum(c["minutes"] for c in chunks), 2),
                     "peak_rss_gb": max(c["peak_rss_gb"] for c in chunks),
                     "peak_mps_driver_gb": max(c["peak_mps_driver_gb"] for c in chunks)}
    out["guard"] = g
    write_results("matched", out)
    print("done")
    if args.figures:
        import figures

        figures.matched()


if __name__ == "__main__":
    main()
