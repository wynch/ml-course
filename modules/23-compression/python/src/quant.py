"""Section 3: int8 and 4-bit weight quantisation of SmolLM2-360M-Instruct, by hand.

Three schemes from src/compress.py, applied to every 2-D weight: the 224
block matrices and the embedding (tied, so it is also the output layer).
Norm weights stay in bf16.

    int8, per row     symmetric, one bf16 scale per row
    4-bit, per row    affine, one bf16 scale and bias per row
    4-bit, group 64   affine, one bf16 scale and bias per 64 weights (MLX's rule)

The quantised weights are put back as float32 and the model runs as before
("fake" quantisation): this measures the damage rounding does, not speed or
memory. Bytes are what storing the integers, scales and biases would take.
Tokens per second are measured for the untouched model and for the 4-bit
group-64 one, to show that fake quantisation makes nothing faster.

Run me:
    uv run python src/quant.py            # numbers, results/quant.json
    uv run python src/quant.py --figures  # also figures/quant.png
"""

from __future__ import annotations

import argparse
import base64
import statistics
import time

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch

from common import Guard, free, get_device, load_hf_model, load_tokenizer, perplexity, ppl_ids, seed_all, write_results
from compress import SCHEMES, quantizable

SPEED_PROMPT = "Write a short paragraph about the history of the bicycle."
SPEED_TOKENS = 128
SPEED_REPEATS = 3
SHOWN = "model.layers.15.mlp.down_proj.weight"   # the matrix behind the figure and the explorable
SHOWN_ROWS, SHOWN_COLS = 32, 512
HIST_EDGES = np.linspace(-0.25, 0.25, 101)


@torch.no_grad()
def tokens_per_second(model, tok) -> dict:
    msgs = [{"role": "user", "content": SPEED_PROMPT}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    kw = dict(max_new_tokens=SPEED_TOKENS, min_new_tokens=SPEED_TOKENS, do_sample=False)
    model.generate(**enc, **kw)                    # warm-up
    rates = []
    for _ in range(SPEED_REPEATS):
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        t0 = time.perf_counter()
        out = model.generate(**enc, **kw)
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        n = out.shape[1] - enc["input_ids"].shape[1]
        rates.append(n / (time.perf_counter() - t0))
    return {"tokens": SPEED_TOKENS, "repeats": SPEED_REPEATS, "median_tok_s": round(statistics.median(rates), 1),
            "all_tok_s": [round(r, 1) for r in rates]}


def b64_bf16(t: torch.Tensor) -> str:
    return base64.b64encode(t.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2").tobytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    guard = Guard("quant.py")
    seed_all()
    device = get_device()
    tok = load_tokenizer()
    ids = ppl_ids()
    model = load_hf_model(str(device))
    guard.tick("load")

    params = list(quantizable(model))
    n_all = sum(p.numel() for p in model.parameters())
    n_q = sum(p.numel() for _, p in params)
    n_rest = n_all - n_q                          # norm weights, kept in bf16
    originals = {n: p.detach().to("cpu", torch.bfloat16).clone() for n, p in params}

    base = perplexity(model, ids)
    speed_base = tokens_per_second(model, tok)
    rows = [{"scheme": "untouched (bf16 checkpoint, run in float32)", **base, "bytes": 2 * n_all,
             "bits_per_weight": 16.0, "mean_rel_error": 0.0, "speed": speed_base}]
    print(f"untouched: ppl {base['ppl']}, {2 * n_all / 1e6:.1f} MB in bf16, {speed_base['median_tok_s']} tok/s")
    guard.tick("untouched")

    W_shown = originals[SHOWN].float()
    shown = {"name": SHOWN, "shape": list(W_shown.shape), "rows": SHOWN_ROWS, "cols": SHOWN_COLS,
             "bf16_b64": b64_bf16(W_shown[:SHOWN_ROWS, :SHOWN_COLS]), "hist_edges": HIST_EDGES.round(4).tolist(),
             "hist": {}, "max_abs": round(float(W_shown.abs().max()), 4), "std": round(float(W_shown.std()), 4)}
    for name, fn in SCHEMES.items():
        bytes_q, errs, per_matrix = 0.0, [], {}
        for n, p in params:
            W = originals[n].to(device).float()
            deq, cost = fn(W)
            p.data.copy_(deq)
            bytes_q += cost["bytes"]
            e = float((deq - W).norm() / W.norm())
            errs.append(e)
            per_matrix[n] = e
            if n == SHOWN:
                d = (deq - W).cpu().numpy().ravel()
                shown["hist"][name] = np.histogram(np.clip(d, HIST_EDGES[0], HIST_EDGES[-1]), HIST_EDGES)[0].tolist()
                shown.setdefault("rel_error", {})[name] = round(e, 5)
        total_bytes = bytes_q + 2 * n_rest
        r = perplexity(model, ids)
        row = {"scheme": name, **r, "bytes": int(total_bytes), "bits_per_weight": round(8 * bytes_q / n_q, 3),
               "mean_rel_error": round(sum(errs) / len(errs), 5),
               "worst_matrix": max(per_matrix, key=per_matrix.get), "worst_rel_error": round(max(errs), 5)}
        if name == "4-bit, group 64":
            row["speed"] = tokens_per_second(model, tok)
        rows.append(row)
        print(f"{name:16s} ppl {r['ppl']:.3f}  {total_bytes / 1e6:.1f} MB  {row['bits_per_weight']} bits/weight  "
              f"mean rel error {row['mean_rel_error']}")
        guard.tick(name)

    out = {
        "model": common.MODEL_ID, "revision": common.MODEL_REVISION, "dtype": "float32", "device": str(device),
        "quantized_weights": n_q, "kept_bf16": n_rest, "all_params": n_all,
        "ppl_setup": {"windows": common.PPL_WINDOWS, "window_tokens": common.PPL_LEN,
                      "predicted_tokens": base["tokens"], "text": "wikitext-103 train, first two row groups"},
        "speed_setup": {"prompt": SPEED_PROMPT, "greedy": True, "new_tokens": SPEED_TOKENS,
                        "repeats": SPEED_REPEATS, "framework": "transformers generate, KV cache, float32 on MPS"},
        "rows": rows, "shown_matrix": shown,
    }
    del model, originals
    free()
    out["guard"] = guard.report()
    write_results("quant", out)
    if args.figures:
        import figures

        figures.quant()


if __name__ == "__main__":
    main()
