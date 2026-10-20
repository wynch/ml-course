"""Logit lens: read the model's mind, layer by layer.

A decoder LLM refines a residual stream through its layers. The "logit lens"
trick: take the hidden state *after each layer*, apply the model's final norm +
output head, and see what token that intermediate state would predict. Early
layers are noise; somewhere in the stack the answer snaps into place and stays.

This module is reusable — exercise (b) imports :func:`layerwise_predictions` to
find the "decision layer" for arbitrary prompts.

Run me:
    uv run python src/logit_lens.py            # print the layer-by-layer table
    uv run python src/logit_lens.py --figure   # draw the convergence heatmap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_model, load_tokenizer

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
RESULTS = Path(__file__).resolve().parents[2] / "results" / "logit_lens.json"

PROMPT = "The Eiffel Tower is located in the city of"
# Exercise (b)'s three prompts, plus the sampling lab's prompt.
DECISION_PROMPTS = [
    PROMPT,
    "The chemical symbol for gold is",
    "Two plus two equals",
    "The capital of France is",
]


@torch.no_grad()
def layerwise_predictions(model, tokenizer, prompt: str):
    """Project every layer's last-position hidden state through the output head.

    Returns a dict with, per layer: the top-1 token string, its probability,
    and the rank the *final* answer holds at that layer (how close the model is
    to committing). Works on any Llama/SmolLM-style model.
    """
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model(**ids, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple: embeddings + one per layer

    # The output head (tied to embeddings) and the final RMSNorm.
    norm = model.model.norm
    head = model.lm_head

    final_top1 = int(out.logits[0, -1].argmax().item())

    rows = []
    last = len(hidden_states) - 1
    for layer_idx, hs in enumerate(hidden_states):
        h = hs[0, -1]  # last position
        # hidden_states[-1] already went through the final norm; do not apply it twice.
        logits = head(h) if layer_idx == last else head(norm(h))
        probs = F.softmax(logits.float(), dim=-1)
        top1 = int(probs.argmax().item())
        # rank of the final answer at this layer (0 = already top-1)
        order = probs.argsort(descending=True)
        rank_final = int((order == final_top1).nonzero()[0, 0].item())
        rows.append(
            {
                "layer": layer_idx,
                "top1_token": tokenizer.decode([top1]),
                "top1_prob": float(probs[top1]),
                "final_token": tokenizer.decode([final_top1]),
                "final_prob": float(probs[final_top1]),
                "final_rank": rank_final,
            }
        )
    return {"rows": rows, "final_token": tokenizer.decode([final_top1])}


def decision_layer(rows) -> int:
    """First layer from which the top-1 token equals the final answer at every later layer."""
    final = rows[-1]["top1_token"]
    for i in range(len(rows)):
        if all(r["top1_token"] == final for r in rows[i:]):
            return rows[i]["layer"]
    return rows[-1]["layer"]


@torch.no_grad()
def save_results(model, tokenizer, out_path: Path = RESULTS) -> dict:
    """Write the lens table for PROMPT and the decision layer of each prompt."""
    res = layerwise_predictions(model, tokenizer, PROMPT)
    ids = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    model_prob = F.softmax(model(**ids).logits[0, -1].float(), dim=-1)
    final_id = int(model_prob.argmax())
    data = {
        "model": model.config.name_or_path,
        "prompt": PROMPT,
        "model_output": {"token": tokenizer.decode([final_id]),
                         "prob": round(float(model_prob[final_id]), 4)},
        "rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                 for r in res["rows"]],
        "decision_layers": {},
    }
    # The bug this file avoids: norming the already-normed last row a second time.
    data["last_row_normed_twice"] = {}
    for prompt in (PROMPT, "The capital of France is"):
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        h = model(**ids, output_hidden_states=True).hidden_states[-1][0, -1]
        p2 = F.softmax(model.lm_head(model.model.norm(h)).float(), dim=-1)
        top = int(p2.argmax())
        data["last_row_normed_twice"][prompt] = {
            "top1_token": tokenizer.decode([top]), "top1_prob": round(float(p2[top]), 4)}
    for prompt in DECISION_PROMPTS:
        r = layerwise_predictions(model, tokenizer, prompt)
        data["decision_layers"][prompt] = {
            "final_token": r["final_token"],
            "final_prob": round(r["rows"][-1]["top1_prob"], 4),
            "decision_layer": decision_layer(r["rows"]),
        }
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {out_path}")
    return data


def fig_logit_lens(model, tokenizer, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    res = layerwise_predictions(model, tokenizer, PROMPT)
    rows = res["rows"]
    ranks = np.array([r["final_rank"] for r in rows], dtype=float)
    # log10(rank+1): 0 means the final answer is already #1 at this layer.
    heat = np.log10(ranks + 1).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(6.5, 9))
    im = ax.imshow(heat, aspect="auto", cmap="viridis_r", vmin=0, vmax=np.log10(1000))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f'{r["layer"]:>2}  →  {r["top1_token"]!r}' for r in rows],
        fontfamily="monospace", fontsize=8,
    )
    ax.set_xticks([])
    ax.set_ylabel("layer  →  its top-1 prediction")
    dl = decision_layer(rows)
    ax.axhline(dl - 0.5, color="crimson", lw=2)
    ax.set_title(
        f'Logit lens: how "{res["final_token"].strip()}" emerges\n'
        f'prompt: "{PROMPT}"\nred line = decision layer {dl}',
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.05)
    cbar.set_label("log10(rank of final answer + 1)   (0 = already #1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    model = load_model()
    tok = load_tokenizer()

    res = layerwise_predictions(model, tok, args.prompt)
    print(f'\nprompt: "{args.prompt}"')
    print(f'final answer: {res["final_token"]!r}\n')
    print(f'{"layer":>5}  {"top-1":<14} {"prob":>7}   {"final-rank":>10}')
    for r in res["rows"]:
        mark = "  <-- locks in" if r["final_rank"] == 0 else ""
        print(f'{r["layer"]:>5}  {r["top1_token"]!r:<14} {r["top1_prob"]:>6.1%}   {r["final_rank"]:>10}{mark}')
    print(f"\ndecision layer: {decision_layer(res['rows'])}")

    if args.figure:
        FIG_DIR.mkdir(exist_ok=True)
        fig_logit_lens(model, tok, FIG_DIR / "logit_lens.png")
        save_results(model, tok)


if __name__ == "__main__":
    main()
