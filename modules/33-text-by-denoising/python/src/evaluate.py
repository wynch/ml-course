"""Section 2: score the trained models on held-out text, and fill a gap in the middle of a line.

Likelihood, per character in nats, on 256 fixed validation blocks of 128:
  ar     exact (the first character from the training frequencies)
  mdm    the ELBO (32 masking rates, two mask seeds), and the exact score of
         three fixed orders: left to right, right to left, a random order per block
  bd4, bd16   the block-diffusion ELBO
  mdm6x  mdm's model trained six times longer: its ELBO and the same three orders

Infill: in each block, hide characters 56-71 and fill them greedily.
  mdm, both sides    the other 112 characters given; 16 steps, most confident first
  mdm, left only     characters 56-127 hidden; 72 steps; only 56-71 are scored
  ar, left only      characters 0-55 given; 16 characters left to right
  (the two mdm rows again for mdm6x)

    uv run python src/evaluate.py        # writes results/evaluate.json, about a minute on MPS
"""

from __future__ import annotations

import json

import common
import torch

import data as D
import diffusion as Df
from train import load_model

SPAN = (56, 72)
M04 = common.MODULE.parent / "04-attention-transformer" / "results" / "train_summary.json"


def infill(models: dict, blocks: torch.Tensor) -> dict:
    M = models["mdm"].cfg.mask_id
    a, b = SPAN
    truth = blocks[:, a:b]
    both = blocks.clone()
    both[:, a:b] = M
    left = blocks.clone()
    left[:, a:] = M
    fills = {
        "mdm_both_sides": Df.sample_mdm(models["mdm"], len(blocks), b - a, temperature=0, x_init=both)[:, a:b],
        "mdm_left_only": Df.sample_mdm(models["mdm"], len(blocks), 128 - a, temperature=0, x_init=left)[:, a:b],
        "mdm6x_both_sides": Df.sample_mdm(models["mdm6x"], len(blocks), b - a, temperature=0, x_init=both)[:, a:b],
        "mdm6x_left_only": Df.sample_mdm(models["mdm6x"], len(blocks), 128 - a, temperature=0, x_init=left)[:, a:b],
        "ar_left_only": Df.continue_ar(models["ar"], blocks[:, :a], b - a),
    }
    out = {}
    for k, f in fills.items():
        f = f.cpu()
        out[k] = {"char_accuracy": round(float((f == truth).float().mean()), 4),
                  "span_exact": round(float((f == truth).all(1).float().mean()), 4),
                  "by_position": [round(float(v), 3) for v in (f == truth).float().mean(0)]}
    return out, {k: v[:6].cpu() for k, v in fills.items()}


def one_char(models: dict, blocks: torch.Tensor) -> dict:
    """Written after the span results: hide one character (position 63) and guess it greedily.
    The diffusion models see the other 127; the AR twin sees the 63 before it."""
    i, M = 63, models["mdm"].cfg.mask_id
    x = blocks.clone()
    x[:, i] = M
    truth = blocks[:, i]
    out = {"ar_left_only": Df.continue_ar(models["ar"], blocks[:, :i], 1)[:, 0].cpu()}
    for name in ("mdm", "mdm6x"):
        with torch.no_grad():
            out[f"{name}_both_sides"] = models[name](x.to(next(models[name].parameters()).device))[:, i].argmax(-1).cpu()
    return {k: round(float((v == truth).float().mean()), 4) for k, v in out.items()}


def main() -> None:
    dev = common.device()
    guard = common.Guard("evaluate", dev)
    tok, train, val = D.load()
    lu = D.log_unigram(train, tok.vocab_size)
    vb = D.val_blocks(val, 256)
    models = {a: load_model(a, dev) for a in ("ar", "mdm", "bd4", "bd16", "mdm6x")}
    like = {
        "ar_exact": round(Df.ar_nll(models["ar"], vb, lu), 4),
        "mdm_elbo": round(Df.elbo(models["mdm"], vb, K=32, seed=0), 4),
        "mdm_elbo_seed1": round(Df.elbo(models["mdm"], vb, K=32, seed=1), 4),
        "bd4_elbo": round(Df.elbo(models["bd4"], vb, K=32, L=4, seed=0), 4),
        "bd16_elbo": round(Df.elbo(models["bd16"], vb, K=32, L=16, seed=0), 4),
        "mdm6x_elbo": round(Df.elbo(models["mdm6x"], vb, K=32, seed=0), 4),
        "unigram": round(float(-lu[vb.flatten()].mean()), 4),
    }
    guard.tick("elbo")
    for name in ("mdm", "mdm6x"):
        for order in ("l2r", "r2l", "random"):
            like[f"{name}_exact_{order}"] = round(Df.ordered_nll(models[name], vb, order), 4)
    guard.tick("ordered")
    fill, examples = infill(models, vb)
    single = one_char(models, vb)
    guard.tick("infill")
    a, b = SPAN
    ex = []
    for i in range(6):
        ex.append({"left": tok.decode(vb[i, a - 24: a].tolist()), "truth": tok.decode(vb[i, a:b].tolist()),
                   "right": tok.decode(vb[i, b: b + 24].tolist()),
                   **{k: tok.decode(v[i].tolist()) for k, v in examples.items()}})
    common.write_results("evaluate", {
        "blocks": {"n": len(vb), "length": vb.size(1), "chars": vb.numel()},
        "likelihood_nats_per_char": like,
        "module04_val_loss": {"value": json.loads(M04.read_text())["final"]["val"],
                              "source": "modules/04-attention-transformer/results/train_summary.json, final val (random windows)"},
        "infill": {"span": [a, b], "results": fill, "examples": ex},
        "one_character": {"position": 63, "written_after_the_span_results": True, "accuracy": single},
        "guard": guard.report(),
    })


if __name__ == "__main__":
    main()
