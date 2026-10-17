"""Section 2: a linear probe, layer by layer, on SmolLM2-360M-Instruct.

A probe asks: is a fact *linearly readable* from the residual stream at this
layer? We feed SST-2 movie-review sentences, keep the residual vector at the
last token after every layer, and fit a logistic regression per layer to
predict positive vs negative.

Two controls keep the probe honest:
  - layer 0 (the token embeddings, before any block runs): what the last
    token alone gives away;
  - the same architecture with random weights: what a probe can read from any
    big random projection of the words. A probe that scores well there is
    measuring itself, not the model.

Run me:
    uv run python src/probe.py            # numbers, results/probe.json
    uv run python src/probe.py --figures  # also figures/probe_by_layer.png
"""

from __future__ import annotations

import argparse
import gc

from common import (FIGURES, MODEL_ID, Guard, free, load_hf_model, load_tokenizer,
                    wrap, write_results)

import numpy as np
import torch

SST2_ID = "stanfordnlp/sst2"
SST2_REVISION = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
N_TRAIN = 2000          # random sample of SST-2 train, seed 0
BATCH = 32
C = 0.1                 # inverse L2 strength; fixed, not tuned on the test set
PLATEAU_POINTS = 1.0    # plateau = first layer within this many points of the best


def load_sst2(n_train: int = N_TRAIN, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset(SST2_ID, revision=SST2_REVISION)
    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    test = ds["validation"]          # the labelled 872; SST-2's test labels are hidden
    clean = lambda xs: [x.strip() for x in xs]
    return (clean(train["sentence"]), np.array(train["label"]),
            clean(test["sentence"]), np.array(test["label"]))


@torch.no_grad()
def last_token_residuals(model, tokenizer, sentences: list[str], batch: int = BATCH) -> np.ndarray:
    """(n, n_layers + 1, d_model): embeddings, then the residual after each block.

    Sentences are left-padded so the last token sits at position -1 in every
    row; position ids come from the mask, so padding does not shift RoPE.
    """
    lm = wrap(model, tokenizer)
    n_layers = model.config.num_hidden_layers
    dev = model.device
    tokenizer.padding_side = "left"
    out = []
    for s in range(0, len(sentences), batch):
        enc = tokenizer(sentences[s : s + batch], return_tensors="pt", padding=True)
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        with lm.trace(enc.input_ids.to(dev), attention_mask=enc.attention_mask.to(dev),
                      position_ids=pos.to(dev)) as tracer:
            hs = [lm.model.embed_tokens.output[:, -1]]
            for i in range(n_layers):
                hs.append(lm.model.layers[i].output[:, -1])
            stacked = torch.stack(hs, 1).float().cpu().save()
            tracer.stop()                # skip the final norm and the 49,152-wide head
        out.append(stacked.numpy())
        del stacked
        gc.collect()                     # stop() leaves the forward's frames in a cycle
    return np.concatenate(out)


def probe_accuracy(Xtr, ytr, Xte, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def accuracy_by_layer(Htr, ytr, Hte, yte) -> list[float]:
    return [round(probe_accuracy(Htr[:, l], ytr, Hte[:, l], yte), 4) for l in range(Htr.shape[1])]


def plateau_layer(acc: list[float], points: float = PLATEAU_POINTS) -> int:
    best = max(acc)
    return next(i for i, a in enumerate(acc) if a >= best - points / 100)


def run(figures: bool = False) -> dict:
    guard = Guard("probe.py")
    tr_x, tr_y, te_x, te_y = load_sst2()
    tok = load_tokenizer()
    guard.tick("load data")

    curves = {}
    for name, random_weights in (("trained", False), ("random_weights", True)):
        model = load_hf_model(random_weights=random_weights)
        Htr = last_token_residuals(model, tok, tr_x)
        Hte = last_token_residuals(model, tok, te_x)
        guard.tick(f"residuals {name}")
        del model
        free()                          # one model in memory at a time
        curves[name] = accuracy_by_layer(Htr, tr_y, Hte, te_y)
        guard.tick(f"probes {name}")
        print(f"{name:>15}: " + " ".join(f"{a:.2f}" for a in curves[name]))

    t = curves["trained"]
    r = curves["random_weights"]
    best = int(np.argmax(t))
    out = {
        "model": MODEL_ID,
        "dataset": f"{SST2_ID}@{SST2_REVISION[:7]}: {N_TRAIN} train sentences (seed 0), "
                   f"{len(te_y)} validation sentences as the test set",
        "probe": f"StandardScaler + LogisticRegression(C={C}), one per layer, last-token residual",
        "majority_baseline": round(float(max(te_y.mean(), 1 - te_y.mean())), 4),
        "layers": list(range(len(t))),
        "accuracy_trained": t,
        "accuracy_random_weights": r,
        "layer0_trained": t[0],
        "best_layer": best,
        "best_accuracy": t[best],
        "plateau_rule": f"first layer within {PLATEAU_POINTS:g} point of the best",
        "plateau_layer": plateau_layer(t),
        "plateau_accuracy": t[plateau_layer(t)],
        "random_best_layer": int(np.argmax(r)),
        "random_best_accuracy": max(r),
        "final_layer_trained": t[-1],
    }
    print(f"best layer {best} at {t[best]:.3f}; plateau from layer {out['plateau_layer']}; "
          f"layer 0 {t[0]:.3f}; random-weight best {max(r):.3f}; majority {out['majority_baseline']}")
    if figures:
        fig_probe(out, FIGURES / "probe_by_layer.png")
    guard.tick("figures")
    out["guard"] = guard.report()
    write_results("probe", out)
    return out


def fig_probe(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = res["layers"]
    t = np.array(res["accuracy_trained"]) * 100
    r = np.array(res["accuracy_random_weights"]) * 100
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(L, t, "-o", ms=3.5, color="#1f918d", label="SmolLM2-360M-Instruct")
    ax.plot(L, r, "-o", ms=3.5, color="#9dada8", label="same architecture, random weights")
    ax.axhline(res["majority_baseline"] * 100, color="#c1121f", ls=":", lw=1,
               label=f"always guess the majority ({res['majority_baseline']*100:.1f}%)")
    p = res["plateau_layer"]
    ax.axvline(p, color="#453781", ls="--", lw=1)
    ax.annotate(f"plateau from layer {p}\n({res['plateau_accuracy']*100:.1f}%)",
                (p, res["plateau_accuracy"] * 100), xytext=(p + 1.2, 62),
                fontsize=9, color="#453781", arrowprops=dict(arrowstyle="->", color="#453781"))
    ax.annotate(f"layer 0 (embeddings): {t[0]:.1f}%", (0, t[0]), xytext=(1.5, t[0] - 6),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xlabel("layer (0 = token embeddings; k = residual after block k)")
    ax.set_ylabel("probe accuracy on 872 held-out sentences (%)")
    ax.set_title("Is sentiment linearly readable at the last token? SST-2, one probe per layer",
                 fontsize=10.5)
    ax.set_ylim(45, 100)
    ax.set_xlim(-0.5, L[-1] + 0.5)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5)
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
