"""Section 3: activation patching, Eiffel Tower against Colosseum.

Two prompts that differ only in the landmark:
    clean   "The Eiffel Tower is located in the city of"  -> ' Paris'
    corrupt "The Colosseum is located in the city of"     -> ' Rome'
Both are 10 tokens, so position k means the same slot in each.

Run the clean prompt once and keep its residual stream. Then run the corrupt
prompt again and again, each time pasting the clean residual into one
(layer, position) cell. If the answer swings back to Paris, that cell carried
the information that decides it.

Metric: logit difference LD = logit(' Paris') - logit(' Rome'), normalised so
that 0 = the corrupt run and 1 = the clean run:
    score = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)

Run me:
    uv run python src/patching.py            # numbers, results/patching.json
    uv run python src/patching.py --figures  # also figures/patching_heatmap.png
"""

from __future__ import annotations

import argparse

from common import (FIGURES, MODEL_ID, Guard, load_hf_model, load_tokenizer, wrap,
                    write_results)

import numpy as np
import torch

CLEAN = "The Eiffel Tower is located in the city of"
CORRUPT = "The Colosseum is located in the city of"
ANSWERS = (" Paris", " Rome")
RESTORE = 0.5   # a cell "restores the answer" when it brings back half the clean LD


def token_ids(tok, text: str) -> torch.Tensor:
    return tok(text, return_tensors="pt").input_ids


def site(lm, row: int):
    """Row 0 = token embeddings; row k (1..32) = residual after the k-th block.

    The same axis as probe.py. ``layers`` counts from 0, so the k-th block is
    ``layers[k - 1]``.
    """
    return lm.model.embed_tokens if row == 0 else lm.model.layers[row - 1]


@torch.no_grad()
def clean_residuals(lm, ids: torch.Tensor, n_rows: int) -> torch.Tensor:
    """(n_rows, seq, d_model) residual stream of one prompt."""
    with lm.trace(ids):
        rows = []
        for r in range(n_rows):
            rows.append(site(lm, r).output[0])
        saved = torch.stack(rows).save()
    return saved


@torch.no_grad()
def logit_diff(lm, ids: torch.Tensor, a: int, b: int) -> float:
    with lm.trace(ids):
        last = lm.lm_head.output[0, -1].save()
    return float(last[a] - last[b])


@torch.no_grad()
def patch_sweep(lm, corrupt_ids: torch.Tensor, clean_resid: torch.Tensor, a: int, b: int) -> np.ndarray:
    """Logit difference after patching each (row, position) cell, (n_rows, seq).

    One trace per row: the corrupt prompt is repeated once per position, and
    batch item p gets the clean residual pasted at position p only.
    """
    n_rows, seq, _ = clean_resid.shape
    batch = corrupt_ids.repeat(seq, 1)
    out = np.zeros((n_rows, seq))
    for r in range(n_rows):
        with lm.trace(batch):
            h = site(lm, r).output
            for p in range(seq):
                h[p, p] = clean_resid[r, p]
            last = lm.lm_head.output[:, -1].save()
        out[r] = (last[:, a] - last[:, b]).float().cpu().numpy()
    return out


@torch.no_grad()
def mps_vs_cpu(model, ids: torch.Tensor) -> dict:
    """One forward pass on MPS, then the same weights moved to CPU."""
    if model.device.type != "mps":
        return {"skipped": "no MPS device"}
    l_mps = model(ids.to("mps")).logits.float().cpu()
    model.to("cpu")
    l_cpu = model(ids).logits.float()
    return {
        "prompt": CLEAN,
        "max_abs_diff_logits": float((l_mps - l_cpu).abs().max()),
        "logit_range": [float(l_cpu.min()), float(l_cpu.max())],
        "same_top1_every_position": bool((l_mps.argmax(-1) == l_cpu.argmax(-1)).all()),
    }


@torch.no_grad()
def nnsight_vs_plain(model, lm, ids: torch.Tensor) -> float:
    """nnsight's wrapped forward against the plain Hugging Face forward."""
    plain = model(ids.to(model.device)).logits
    with lm.trace(ids.to(model.device)):
        traced = lm.lm_head.output.save()
    return float((plain - traced).abs().max())


def run(figures: bool = False) -> dict:
    guard = Guard("patching.py")
    tok = load_tokenizer()
    model = load_hf_model()
    lm = wrap(model, tok)
    dev = model.device
    clean = token_ids(tok, CLEAN).to(dev)
    corrupt = token_ids(tok, CORRUPT).to(dev)
    assert clean.shape == corrupt.shape, (clean.shape, corrupt.shape)
    a, b = (tok(x, add_special_tokens=False).input_ids for x in ANSWERS)
    assert len(a) == 1 and len(b) == 1, "each answer must be one token"
    a, b = a[0], b[0]
    n_rows = model.config.num_hidden_layers + 1
    guard.tick("load")

    ld_clean = logit_diff(lm, clean, a, b)
    ld_corrupt = logit_diff(lm, corrupt, a, b)
    resid = clean_residuals(lm, clean, n_rows)
    raw = patch_sweep(lm, corrupt, resid, a, b)
    score = (raw - ld_corrupt) / (ld_clean - ld_corrupt)
    guard.tick("sweep")

    seq = clean.shape[1]
    toks_clean = [tok.decode([int(t)]) for t in clean[0]]
    toks_corrupt = [tok.decode([int(t)]) for t in corrupt[0]]
    differ = [p for p in range(seq) if toks_clean[p] != toks_corrupt[p]]
    last = seq - 1
    # Where does the answer live? Two places to read off the heatmap:
    subj_best = max(((r, p) for r in range(n_rows) for p in differ), key=lambda rp: score[rp])
    last_rows = [r for r in range(n_rows) if score[r, last] >= RESTORE]
    subj_rows = [r for r in range(n_rows) if max(score[r, p] for p in differ) >= RESTORE]
    out = {
        "model": MODEL_ID,
        "clean": CLEAN,
        "corrupt": CORRUPT,
        "answers": list(ANSWERS),
        "tokens_clean": toks_clean,
        "tokens_corrupt": toks_corrupt,
        "differing_positions": differ,
        "ld_clean": round(ld_clean, 3),
        "ld_corrupt": round(ld_corrupt, 3),
        "rows": "0 = token embeddings, k = residual after block k",
        "score": [[round(float(v), 3) for v in row] for row in score],
        "restore_threshold": RESTORE,
        "best_subject_cell": {"row": subj_best[0], "position": subj_best[1],
                              "token": toks_clean[subj_best[1]],
                              "score": round(float(score[subj_best]), 3)},
        "subject_rows_restoring": subj_rows,
        "last_position_rows_restoring": last_rows,
        "last_position_first_restoring_row": last_rows[0] if last_rows else None,
    }
    print("tokens:", toks_clean, "vs", toks_corrupt, "differ at", differ)
    print(f"LD clean {ld_clean:.2f}, corrupt {ld_corrupt:.2f}")
    print(f"best subject cell: row {subj_best[0]} pos {subj_best[1]} {toks_clean[subj_best[1]]!r} "
          f"score {score[subj_best]:.2f}; subject rows >= {RESTORE}: {subj_rows}")
    print(f"last position rows >= {RESTORE}: {last_rows}")

    out["nnsight_vs_plain_max_abs_diff"] = nnsight_vs_plain(model, lm, clean.cpu())
    out["mps_vs_cpu"] = mps_vs_cpu(model, clean.cpu())
    print(f"nnsight vs plain forward: {out['nnsight_vs_plain_max_abs_diff']:.2e}; "
          f"MPS vs CPU: {out['mps_vs_cpu']}")
    guard.tick("checks")
    if figures:
        fig_patching(out, FIGURES / "patching_heatmap.png")
    guard.tick("figures")
    out["guard"] = guard.report()
    write_results("patching", out)
    return out


def fig_patching(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = np.array(res["score"])
    n_rows, seq = S.shape
    fig, ax = plt.subplots(figsize=(7.4, 9.2))
    lim = max(1.0, float(np.abs(S).max()))
    im = ax.imshow(S, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    labels = [f"{c!r}\n{k!r}" if c != k else repr(c)
              for c, k in zip(res["tokens_clean"], res["tokens_corrupt"])]
    ax.set_xticks(range(seq))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=0, fontfamily="monospace")
    ax.set_yticks(range(0, n_rows, 2))
    ax.set_yticklabels(["emb"] + [str(r) for r in range(2, n_rows, 2)], fontsize=8)
    ax.set_ylabel("layer whose output we patch (emb = token embeddings)")
    ax.set_xlabel("position (top: clean token, bottom: corrupt token)")
    b = res["best_subject_cell"]
    ax.add_patch(plt.Rectangle((b["position"] - 0.5, b["row"] - 0.5), 1, 1, fill=False, ec="k", lw=1.6))
    f = res["last_position_first_restoring_row"]
    if f is not None:
        ax.add_patch(plt.Rectangle((seq - 1.5, f - 0.5), 1, 1, fill=False, ec="k", lw=1.6, ls="--"))
    ax.set_title(
        "Patch one clean residual cell into the corrupt run\n"
        f"score 1 = Paris restored (LD {res['ld_clean']:+.2f}), 0 = Rome as corrupt (LD {res['ld_corrupt']:+.2f})",
        fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.045)
    cb.set_label("normalised logit difference  ' Paris' - ' Rome'")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    run(figures=args.figures)


if __name__ == "__main__":
    main()
