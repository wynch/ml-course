"""SVD data for the explorable 06-lora-rank.html, from this module's own files.

Two matrices from the same layer, so the comparison is fair:
  W0  the frozen SmolLM2-360M-Instruct weight  model.layers.0.self_attn.q_proj
  dW  the trained update (alpha/r)*B*A of the exercise (a) r=32 adapter on
      that same projection
Both sliced to [:48, :48] so the browser can paint them. Writes
results/svd_explorable.json (the numbers the explorable and quiz quote) and,
with --splice, replaces the first two entries of the explorable's SVD array.

Run:  uv run python svd_explorable.py [--splice]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from src.common import MODEL_ID

HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "outputs" / "ex-a-r32" / "adapter"
EXPLORABLE = HERE.parents[2] / "explorables" / "06-lora-rank.html"
N = 48
KEY = "model.layers.0.self_attn.q_proj"


def entry(W: np.ndarray, label: str, note: str, source: str) -> dict:
    U, S, Vt = np.linalg.svd(W.astype(np.float64))
    total = float((S ** 2).sum())
    tail = np.concatenate([np.cumsum((S ** 2)[::-1])[::-1], [0.0]])
    err = np.sqrt(np.maximum(tail, 0) / total)
    energy = 1 - tail / total
    return {
        "label": label, "note": note, "source": source, "n": N, "m": N,
        "U": [round(float(x), 6) for x in U.T.ravel()],
        "S": [float(f"{s:.9g}") for s in S],
        "V": [round(float(x), 6) for x in Vt.ravel()],
        "absmax": round(float(np.abs(W).max()), 6),
        "errRef": [round(float(x), 6) for x in err],
        "energy": [round(float(x), 6) for x in energy],
    }


def summary(e: dict) -> dict:
    err = e["errRef"]
    return {
        "rank_for_20pct_error": next(r for r, x in enumerate(err) if x <= 0.20),
        "err_r8": err[8], "err_r16": err[16], "err_r32": err[32],
        "energy_r1": e["energy"][1], "energy_r8": e["energy"][8],
    }


def main():
    with safe_open(hf_hub_download(MODEL_ID, "model.safetensors"), "pt") as f:
        W0 = f.get_tensor(f"{KEY}.weight")[:N, :N].float().numpy()

    cfg = json.loads((ADAPTER / "adapter_config.json").read_text())
    with safe_open(str(ADAPTER / "adapter_model.safetensors"), "pt") as f:
        ad = {k: f.get_tensor(k).float().numpy() for k in f.keys()}
    A = next(v for k, v in ad.items() if "layers.0.self_attn.q_proj.lora_A" in k)
    B = next(v for k, v in ad.items() if "layers.0.self_attn.q_proj.lora_B" in k)
    scale = cfg["lora_alpha"] / cfg["r"]
    dW_full = scale * (B.astype(np.float64) @ A.astype(np.float64))
    dW = dW_full[:N, :N]

    e0 = entry(W0, "W₀ — real frozen weight",
               f"SmolLM2-360M-Instruct, layers.0.self_attn.q_proj.weight[:48,:48]",
               f"{MODEL_ID} model.safetensors")
    e1 = entry(dW, "ΔW — real trained LoRA update",
               "module 06 r=32 adapter, the same q_proj: (α/r)·B·A, slice [:48,:48]",
               "modules/06-fine-tuning/python/outputs/ex-a-r32/adapter/adapter_model.safetensors")
    res = {
        "W0": summary(e0), "dW": summary(e1),
        "dW_full_shape": list(dW_full.shape),
        "dW_full_numpy_rank": int(np.linalg.matrix_rank(dW_full)),
        "alpha_over_r": scale,
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "svd_explorable.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))

    if "--splice" in sys.argv:
        html = EXPLORABLE.read_text()
        m = re.search(r"const SVD = (\[.*?\]);\n", html, re.S)
        old = json.loads(m.group(1))
        new = [e0, e1] + old[2:]
        html = html[: m.start(1)] + json.dumps(new, separators=(",", ":")) + html[m.end(1):]
        EXPLORABLE.write_text(html)
        print(f"spliced {EXPLORABLE.name}")


if __name__ == "__main__":
    main()
