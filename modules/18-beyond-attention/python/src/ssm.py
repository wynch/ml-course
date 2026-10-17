"""Section 2: a diagonal linear recurrence, then a selective one, in plain PyTorch.

Both models are five Mamba-style blocks (src/models.py) around the same shell:
in_proj, a short causal convolution, the recurrence, a gate, out_proj. Only the
recurrence differs.

- ``lti``: every channel keeps 16 states, each with its own fixed decay. The
  decays and input weights are the same at every step, so the layer is a
  convolution with a fixed kernel, and training uses that form.
- ``selective``: the step size, B and C come from each input (Mamba-2's form:
  one decay per head and step). Training uses the masked-matrix form.

Both train with the module's one recipe for the same number of steps as the
GRU of section 1. For each model the script also checks that one-token steps
(the recurrence) give the same logits as the parallel training form.

Run me:
    uv run python src/ssm.py            # numbers, results/ssm.json
    uv run python src/ssm.py --figures  # also figures/ssm.png
"""

from __future__ import annotations

import argparse

from common import FIGURES, CharData, Guard, get_device, read_results, write_results  # MPS guard before torch

import torch

from models import build, n_params
from train import sample, train

STEPS = 1000


@torch.no_grad()
def step_vs_parallel(model, data: CharData, device) -> float:
    """Max |logit difference| between the parallel forward and 128 one-token steps."""
    x, _ = data.val_windows(1, batch=2)[0]
    x = x.to(device)
    full, _ = model(x)
    state = model.init_state(2, device)
    outs = []
    for t in range(x.shape[1]):
        lo, state = model.step(x[:, t], state)
        outs.append(lo)
    return (torch.stack(outs, 1) - full).abs().max().item()


def run(steps: int = STEPS) -> dict:
    guard = Guard("ssm")
    device = get_device()
    data = CharData()
    val_full = data.val_windows(100)
    out = {"device": str(device), "gpt_params": 619776, "models": {}}
    for kind in ("lti", "selective"):
        torch.manual_seed(0)
        model = build(kind, data.vocab_size)
        res = train(model, data, device, steps=steps, val_full=val_full)
        guard.tick(kind)
        diff = step_vs_parallel(model, data, device)
        state = model.init_state(1, device)
        text = sample(model, data, device, n=200)
        b = model.blocks[0]
        out["models"][kind] = {
            "params": n_params(model),
            "d_model": model.emb.embedding_dim, "n_layer": len(model.blocks), "E": b.E, "N": b.N,
            "heads": getattr(b, "H", None),
            "state_floats": model.state_floats(state),
            "step_vs_parallel_max_abs_diff": diff,
            "run": res,
            "sample": text,
        }
        print(f"{kind}: {n_params(model):,} parameters, {res['steps']} steps in {res['train_seconds']} s "
              f"({res['ms_per_step']} ms/step), val {res['val_loss']}, step vs parallel {diff:.1e}")
        print(text)
        del model
    out["guard"] = guard.report()
    return out


def fig_ssm(r: dict, path) -> None:
    import matplotlib.pyplot as plt

    gru = read_results("gru")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    series = [("lti", "diagonal linear recurrence", "#9aa5a2", r["models"]["lti"]),
              ("selective", "selective recurrence", "#b7410e", r["models"]["selective"]),
              ("gru", "GRU (section 1)", "#d99a3e", gru)]
    for _, label, color, m in series:
        log = m["run"]["log"]
        ax.plot(log["val_step"], log["val"], "o-", ms=3.5, color=color,
                label=f"{label}: {m['run']['val_loss']:.3f}")
    ax.axhline(gru["committed_gpt_val_loss"], color="#453781", ls="--", lw=1.2,
               label=f"module 04's GPT, 5,000 steps: {gru['committed_gpt_val_loss']:.3f}")
    ax.set_xlabel("training step (64 × 128 characters)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_ylim(1.3, 3.5)
    ax.set_title("Same shell, two recurrences: fixed decays against input-chosen ones", fontsize=10)
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
    write_results("ssm", r)
    if args.figures:
        fig_ssm(r, FIGURES / "ssm.png")


if __name__ == "__main__":
    main()
