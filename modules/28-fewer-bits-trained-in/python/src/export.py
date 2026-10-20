"""Section 2: the bf16 cast at export, and the fine-tune it can undo.

A ternary model trains with float "latent" weights; the file holds only the
codes and one scale per matrix. Sahai et al. (2026, arXiv:2610.07853) found
that real export tools first cast the latents to bf16. bf16 keeps 8 bits of
mantissa, so a latent close to the threshold 0.5 * mean|W| can land on the
other side, or exactly on it, where round-half-to-even gives 0. Small
fine-tuning updates move latents by less than one bf16 step, so the cast can
undo them.

The experiment:
1. Retrain the best 0.5 MB ternary model of section 1 (its shape and rate,
   seed 0), or reload it from the cache.
2. Fine-tune it on tiny-shakespeare (module 04's text, new to this model) for
   FT_STEPS steps at four constant rates, AdamW without weight decay.
3. For each rate, export twice (codes from fp32 latents, codes from bf16-cast
   latents) and count: codes the fine-tune changed (fp32 export after against
   before), and how many of those the bf16 export sets back to their old value.
   Loss on held-out Shakespeare before, after with the fp32 export, after with
   the bf16 export.

Run me (after matched.py is done):
    uv run python src/export.py            # numbers, results/export.json
    uv run python src/export.py --figures  # also figures/export.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch
import torch.nn.functional as F

from bitlinear import block_linears, ternary_codes
from chars import splits
from common import M04, Guard, get_device, read_results, write_results
from matched import export_loss
from train import BLOCK, held_loss, train

FT_RATES = (1e-6, 1e-5, 1e-4, 1e-3)
FT_STEPS = 200
FT_BATCH = 32
FT_SEED = 7
SHAKES_HELD = 100_000


def cache_path(d: int, L: int, lr: float) -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "mlcourse" / "m28" / f"ternary-d{d}-L{L}-lr{lr:g}-s0.pt"


def shakespeare(vocab) -> tuple[np.ndarray, np.ndarray, int]:
    """Module 04's text in this module's character ids: the first 90% to fine-tune, then 100,000 held out."""
    text = (M04 / "corpus" / "tiny_shakespeare.txt").read_text()
    ids = vocab.encode(text)
    n_val = int(len(ids) * 0.1)
    return ids[:-n_val], ids[-n_val:][:SHAKES_HELD], int((ids == vocab.unk).sum())


@torch.no_grad()
def codes(model, bf16: bool) -> dict[str, torch.Tensor]:
    out = {}
    for name, mod in block_linears(model):
        W = mod.weight.to(torch.bfloat16) if bf16 else mod.weight
        out[name] = ternary_codes(W).float().cpu()
    return out


@torch.no_grad()
def ulp_moves(before: dict, model) -> dict:
    """How far the fine-tune moved each latent, in bf16 steps at that latent's size (median and 90th percentile)."""
    moves = []
    for name, mod in block_linears(model):
        W0, W1 = before[name], mod.weight.detach().cpu()
        exp = torch.floor(torch.log2(W0.abs().clamp_min(1e-30)))
        ulp = torch.pow(2.0, exp - 7)                         # bf16 keeps 7 stored mantissa bits
        moves.append(((W1 - W0).abs() / ulp).flatten())
    m = torch.cat(moves)
    return {"median": round(float(m.median()), 3), "p90": round(float(m.quantile(0.9)), 3)}


def finetune(model, data: np.ndarray, lr: float, device) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    t = torch.from_numpy(data)
    g = torch.Generator().manual_seed(FT_SEED)
    starts = torch.randint(len(t) - BLOCK - 1, (FT_STEPS, FT_BATCH), generator=g)
    offs = torch.arange(BLOCK + 1)
    model.train()
    for step in range(FT_STEPS):
        chunk = t[starts[step][:, None] + offs].to(device)
        logits, _ = model(chunk[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    guard = Guard("export.py")
    device = get_device()
    train_ids, tune, report, vocab = splits()
    cell = next(c for c in read_results("matched")["cells"] if c["arm"] == "ternary" and c["budget_mb"] == 0.5)
    d, L, lr = cell["best"]["d"], cell["best"]["n_layer"], cell["best"]["lr"]
    path = cache_path(d, L, lr)
    if path.exists():
        from train import build

        model = build("ternary", d, L, vocab.size).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        retrained = False
    else:
        model, _ = train("ternary", d, L, lr, 0, train_ids, vocab.size, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        retrained = True
    guard.tick("base")
    base_report = round(held_loss(model, report, device), 5)
    # MPS training is not bit-reproducible, so a retrained copy scores near, not exactly at, section 1's seed 0.

    sh_train, sh_held, n_unk = shakespeare(vocab)
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base_latents = {n: m.weight.detach().cpu().clone() for n, m in block_linears(model)}
    c_before = codes(model, bf16=False)
    sh_before = round(held_loss(model, sh_held, device), 5)
    sh_before_bf16 = export_loss(model, sh_held, sh_held, device)["report"]
    n_codes = sum(c.numel() for c in c_before.values())
    cast_flips_before = sum(int((c_before[k] != v).sum()) for k, v in codes(model, bf16=True).items())

    rows = []
    for ft_lr in FT_RATES:
        model.load_state_dict(base_state)
        finetune(model, sh_train, ft_lr, device)
        c32, c16 = codes(model, bf16=False), codes(model, bf16=True)
        changed = sum(int((c32[k] != c_before[k]).sum()) for k in c32)
        reverted = sum(int(((c32[k] != c_before[k]) & (c16[k] == c_before[k])).sum()) for k in c32)
        cast_flips = sum(int((c32[k] != c16[k]).sum()) for k in c32)
        after32 = round(held_loss(model, sh_held, device), 5)
        after16 = export_loss(model, sh_held, sh_held, device)["report"]
        row = {"lr": ft_lr, "codes_changed": changed, "share_changed": round(changed / n_codes, 6),
               "changed_then_reverted_by_bf16": reverted,
               "share_reverted": round(reverted / changed, 4) if changed else None,
               "bf16_cast_flips_after": cast_flips,
               "shakespeare_loss_fp32_export": after32, "shakespeare_loss_bf16_export": after16,
               "gain_fp32": round(sh_before - after32, 5), "gain_bf16": round(sh_before - after16, 5),
               "latent_move_in_bf16_steps": ulp_moves(base_latents, model)}
        rows.append(row)
        print(f"lr {ft_lr:g}: {changed} codes changed, {reverted} reverted by bf16 ({row['share_reverted']}); "
              f"Shakespeare loss {sh_before:.4f} -> fp32 {after32:.4f}, bf16 {after16:.4f}")
        guard.tick(f"ft{ft_lr:g}")

    out = {
        "base": {"d": d, "n_layer": L, "lr": lr, "seed": 0, "retrained_this_run": retrained,
                 "wikitext_report_loss": base_report, "section1_seed0_report": cell["report_by_seed"][0],
                 "codes": n_codes,
                 "bf16_cast_flips": cast_flips_before, "shakespeare_loss": sh_before,
                 "shakespeare_loss_bf16_export": sh_before_bf16},
        "finetune": {"steps": FT_STEPS, "batch": FT_BATCH, "seed": FT_SEED, "rates": list(FT_RATES),
                     "optimizer": "AdamW, betas (0.9, 0.95), no weight decay, constant rate",
                     "shakespeare_unknown_chars": n_unk, "held_chars": len(sh_held)},
        "rows": rows,
    }
    out["guard"] = guard.report()
    write_results("export", out)
    if args.figures:
        import figures

        figures.export()


if __name__ == "__main__":
    main()
