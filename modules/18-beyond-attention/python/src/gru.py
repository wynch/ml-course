"""Section 1: a GRU char model on module 04's data, at the tiny GPT's size.

Two GRU layers of width 224, an embedding and a head tied to it: 619,808
parameters, against the tiny GPT's 619,776. Trained with the module's one
recipe (src/train.py) for a fixed number of steps; the validation loss is
measured on 100 fixed batches of module 04's validation split.

As a yardstick, the same 100 batches go through module 04's committed GPT
(5,000 steps at peak learning rate 3e-4, dropout 0.1). It trained longer and
under another recipe, so it is a reference, not a fair match; section 3 is the
fair match.

Run me:
    uv run python src/gru.py            # numbers, results/gru.json
    uv run python src/gru.py --figures  # also figures/gru.png
"""

from __future__ import annotations

import argparse

from common import FIGURES, CharData, Guard, get_device, write_results  # MPS guard before torch

import torch

from models import build, load_committed_gpt, n_params
from train import evaluate, sample, train

STEPS = 1000


def run(steps: int = STEPS) -> dict:
    guard = Guard("gru")
    device = get_device()
    data = CharData()
    val_full = data.val_windows(100)

    gpt = load_committed_gpt().to(device)
    gpt_val = evaluate(gpt, val_full, device)
    del gpt
    guard.tick("committed gpt")

    torch.manual_seed(0)
    model = build("gru", data.vocab_size)
    res = train(model, data, device, steps=steps, val_full=val_full)
    guard.tick("train")
    text = sample(model, data, device, n=200)
    print(f"GRU: {n_params(model):,} parameters, {res['steps']} steps in {res['train_seconds']} s, "
          f"val {res['val_loss']}  (module 04's committed GPT: {gpt_val:.4f})")
    print(text)
    return {
        "device": str(device),
        "params": n_params(model),
        "gpt_params": 619776,
        "hidden": model.hidden, "n_layer": model.n_layer,
        "state_floats": model.n_layer * model.hidden,
        "committed_gpt_val_loss": round(gpt_val, 4),
        "run": res,
        "sample": text,
        "guard": guard.report(),
    }


def fig_gru(r: dict, path) -> None:
    import matplotlib.pyplot as plt

    log = r["run"]["log"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(log["step"], log["train"], color="#b7410e", lw=1, alpha=0.5, label="GRU, train (25-step mean)")
    ax.plot(log["val_step"], log["val"], "o-", color="#b7410e", ms=4, label="GRU, validation")
    ax.axhline(r["committed_gpt_val_loss"], color="#453781", ls="--", lw=1.2,
               label=f"module 04's GPT, 5,000 steps: {r['committed_gpt_val_loss']:.3f}")
    ax.set_xlabel("training step (64 × 128 characters)")
    ax.set_ylabel("next-character loss (nats)")
    ax.set_ylim(1.3, 3.5)
    ax.set_title(f"A GRU of {r['params']:,} parameters on tiny-shakespeare: "
                 f"validation {r['run']['val_loss']:.3f} after {r['run']['steps']:,} steps", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    r = run(args.steps)
    write_results("gru", r)
    if args.figures:
        fig_gru(r, FIGURES / "gru.png")


if __name__ == "__main__":
    main()
