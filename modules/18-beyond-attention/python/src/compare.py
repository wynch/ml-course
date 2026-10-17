"""Section 3, first half: GPT, GRU and selective SSM for the same wall time.

Each model trains from scratch with the module's one recipe for ``--budget``
seconds of training time on this Mac (evaluation not counted), and the
cosine schedule follows elapsed time, so each gets a full schedule. The GPT
is module 04's architecture, retrained here under the same budget and recipe
with no dropout. Module 04's committed weights (5,000 steps, another recipe)
are scored on the same validation batches as a reference only.

Equal wall time is a fair test of one question: which model learns most on
this machine in this time. It favours whatever runs fastest here, which
depends on the kernels PyTorch has for this chip, not only on the idea.

Run me:
    uv run python src/compare.py               # --budget 60 by default
    uv run python src/compare.py --budget 30 --figures
"""

from __future__ import annotations

import argparse

from common import FIGURES, CharData, Guard, get_device, write_results  # MPS guard before torch

import torch

from models import GPTCache, build, load_committed_gpt, n_params
from train import evaluate, sample, train

BUDGET = 60.0
KINDS = ("gpt", "gru", "selective")
LABELS = {"gpt": "GPT (attention)", "gru": "GRU", "selective": "selective SSM"}
COLORS = {"gpt": "#453781", "gru": "#d99a3e", "selective": "#b7410e"}


def run(budget: float = BUDGET) -> dict:
    guard = Guard("compare")
    device = get_device()
    data = CharData()
    val_full = data.val_windows(100)
    committed = evaluate(load_committed_gpt().to(device), val_full, device)
    out = {"device": str(device), "budget_seconds": budget, "committed_gpt_val_loss": round(committed, 4),
           "models": {}}
    for kind in KINDS:
        torch.manual_seed(0)
        model = build(kind, data.vocab_size)
        res = train(model, data, device, seconds=budget, val_full=val_full)
        guard.tick(kind)
        stepper = GPTCache(model) if kind == "gpt" else model
        text = sample(stepper, data, device, n=120)       # GPT positions stop at 128
        out["models"][kind] = {"params": n_params(model), "run": res, "sample": text}
        print(f"{kind}: {res['steps']} steps in {res['train_seconds']} s ({res['ms_per_step']} ms/step, "
              f"{res['train_tokens_per_s']:,} tokens/s), val {res['val_loss']}")
        del model, stepper
    out["guard"] = guard.report()
    return out


def fig_compare(r: dict, path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for kind in KINDS:
        m = r["models"][kind]
        log = m["run"]["log"]
        ax.plot(log["val_seconds"], log["val"], "o-", ms=3.5, color=COLORS[kind],
                label=f"{LABELS[kind]}: {m['run']['val_loss']:.3f} after {m['run']['steps']:,} steps")
    ax.axhline(r["committed_gpt_val_loss"], color="#453781", ls=":", lw=1.2,
               label=f"module 04's committed GPT (5,000 steps): {r['committed_gpt_val_loss']:.3f}")
    ax.set_xlabel("training seconds on this Mac")
    ax.set_ylabel("validation loss (nats)")
    ax.set_ylim(1.3, 3.5)
    ax.set_title(f"Same budget, {r['budget_seconds']:.0f} s each, same recipe", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=BUDGET, help="training seconds per model")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    r = run(args.budget)
    write_results("compare", r)
    if args.figures:
        fig_compare(r, FIGURES / "compare.png")


if __name__ == "__main__":
    main()
