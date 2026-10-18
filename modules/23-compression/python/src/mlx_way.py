"""Section 4, the library way: MLX's own 4-bit, beside the hand version.

The cached `mlx-community/SmolLM2-360M-Instruct` is the bf16 model converted
to MLX, not a quantised one. So we quantise it here with mlx-lm's own
command (`mlx_lm.convert`, 4 bits, group 64), once, into a folder beside the
Hugging Face cache, and read both copies back with `mlx_lm.load`.

Same measure as sections 2 and 3: the same token ids (from the Hugging Face
tokenizer), the same 32 windows of 1024 tokens, the same mean next-token loss
in float32. Same speed test as section 3: the same prompt ids, 128 greedy
tokens with the end-of-text token blocked, a warm-up then the median of three.

Known-answer check: the hand rule in src/compress.py, run on the Hugging
Face checkpoint's bf16 weights, should give the very weights MLX stored.

Run me:
    uv run python src/mlx_way.py            # numbers, results/mlx_way.json
    uv run python src/mlx_way.py --figures  # also figures/mlx_way.png
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

import common  # noqa: F401  (sets the guard env before torch; torch only runs on the CPU here)
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from common import MLX_ID, MLX_REVISION, Guard, load_tokenizer, ppl_ids, read_results, seed_all, write_results
from compress import quantize_affine

BITS, GROUP = 4, 64


def converted_path() -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "mlcourse" / f"smollm2-360m-instruct-mlx-{MLX_REVISION[:7]}-{BITS}bit-g{GROUP}"


def convert_once() -> tuple[Path, bool]:
    import shutil

    import mlx_lm.utils
    from mlx_lm import convert

    path = converted_path()
    if (path / "model.safetensors").exists() and (path / "config.json").exists():
        return path, False
    # The model card step fetches the repo's README.md, which the cache does not
    # hold; the card is metadata only, so skip it and convert fully offline.
    mlx_lm.utils.create_model_card = lambda *a, **k: None
    tmp = path.with_name(path.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    convert(MLX_ID, mlx_path=str(tmp), quantize=True, q_group_size=GROUP, q_bits=BITS, revision=MLX_REVISION)
    tmp.rename(path)
    return path, True


def perplexity(model, ids: torch.Tensor, batch: int = 2) -> dict:
    total, count = 0.0, 0
    for i in range(0, len(ids), batch):
        x = mx.array(ids[i : i + batch].numpy())
        logits = model(x).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits[:, :-1], x[:, 1:], reduction="sum")
        total += float(nll.item())
        count += int(x[:, 1:].size)
    loss = total / count
    return {"loss": round(loss, 5), "ppl": round(float(np.exp(loss)), 4), "tokens": count}


def tokens_per_second(model, mlx_tok, prompt_ids: list[int], eos_ids: list[int], new_tokens: int,
                      repeats: int) -> dict:
    from mlx_lm import stream_generate

    vocab = model.model.embed_tokens.weight.shape[0]
    mask = np.zeros(vocab, dtype=np.float32)
    mask[eos_ids] = -np.inf
    mask = mx.array(mask)

    def block_eos(_tokens, logits):
        return logits + mask.astype(logits.dtype)

    def run():
        last = None
        for last in stream_generate(model, mlx_tok, prompt_ids, max_tokens=new_tokens, logits_processors=[block_eos]):
            pass
        return last

    run()                                                  # warm-up
    outs = [run() for _ in range(repeats)]
    assert all(o.generation_tokens == new_tokens for o in outs), [o.generation_tokens for o in outs]
    rates = [o.generation_tps for o in outs]
    return {"tokens": new_tokens, "repeats": repeats, "median_tok_s": round(statistics.median(rates), 1),
            "all_tok_s": [round(r, 1) for r in rates]}


def check_against_hand(model) -> dict:
    """Dequantise every MLX weight and compare with the hand rule on the HF checkpoint (both in bf16)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    f = safe_open(hf_hub_download(common.MODEL_ID, "model.safetensors", revision=common.MODEL_REVISION), "pt")
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    n, same, worst = 0, 0, 0.0
    for key in f.keys():
        stem = key[: -len(".weight")]
        if f"{stem}.scales" not in flat:
            continue
        W = f.get_tensor(key)
        mine = quantize_affine(W.float(), BITS, GROUP)[0].to(torch.bfloat16).float().numpy()
        theirs = np.array(mx.dequantize(flat[f"{stem}.weight"], flat[f"{stem}.scales"], flat[f"{stem}.biases"],
                                        group_size=GROUP, bits=BITS).astype(mx.float32))
        d = float(np.abs(mine - theirs).max())
        n += 1
        same += d == 0.0
        worst = max(worst, d)
    return {"matrices": n, "bit_identical": same, "max_abs_diff": worst}


def same_as_hf(mlx_tok_ids: list[int], hf_ids: list[int]) -> dict:
    """Is the cached MLX copy the HF checkpoint? Compare every tensor, and the token ids of our text."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    a = safe_open(hf_hub_download(common.MODEL_ID, "model.safetensors", revision=common.MODEL_REVISION), "pt")
    b = safe_open(hf_hub_download(MLX_ID, "model.safetensors", revision=MLX_REVISION), "pt")
    keys = sorted(set(a.keys()) & set(b.keys()))
    equal = sum(torch.equal(a.get_tensor(k), b.get_tensor(k)) for k in keys)
    return {"hf_tensors": len(list(a.keys())), "mlx_tensors": len(list(b.keys())), "equal_tensors": int(equal),
            "text_tokens": len(hf_ids), "same_token_ids": mlx_tok_ids == hf_ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    from mlx_lm import load

    guard = Guard("mlx_way.py")
    seed_all()
    mx.random.seed(common.SEED)
    tok = load_tokenizer()
    ids = ppl_ids()
    hand = read_results("quant")
    speed = hand["speed_setup"]
    prompt_ids = tok.apply_chat_template([{"role": "user", "content": speed["prompt"]}], add_generation_prompt=True)
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids["input_ids"]
    eos_ids = sorted({tok.eos_token_id, *([tok.pad_token_id] if tok.pad_token_id is not None else [])})

    from corpus import fetch_text
    from transformers import AutoTokenizer

    text, _ = fetch_text()
    mlx_tok_ids = AutoTokenizer.from_pretrained(MLX_ID, revision=MLX_REVISION)(text)["input_ids"]
    copy_check = same_as_hf(mlx_tok_ids, tok(text)["input_ids"])
    print(f"cached MLX copy against the HF checkpoint: {copy_check}")
    guard.tick("copy_check")

    path, converted_now = convert_once()
    guard.tick("convert")

    rows = []
    for label, src, kw in (("MLX bf16 (cached, untouched)", MLX_ID, {"revision": MLX_REVISION}),
                           ("MLX 4-bit, group 64 (mlx_lm.convert)", str(path), {})):
        mx.reset_peak_memory()
        model, mlx_tok = load(src, **kw)
        r = perplexity(model, ids)
        s = tokens_per_second(model, mlx_tok, prompt_ids, eos_ids, speed["new_tokens"], speed["repeats"])
        row = {"setting": label, **r, "speed": s, "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2)}
        if "4-bit" in label:
            row["check_vs_hand"] = check_against_hand(model)
        rows.append(row)
        print(f"{label:38s} ppl {r['ppl']:.3f}  {s['median_tok_s']} tok/s")
        del model
        mx.clear_cache()
        guard.tick(label)

    from huggingface_hub import hf_hub_download

    files = {"bf16": Path(hf_hub_download(MLX_ID, "model.safetensors", revision=MLX_REVISION)).stat().st_size,
             "4bit_g64": sum(p.stat().st_size for p in path.glob("*.safetensors"))}
    out = {
        "mlx_id": MLX_ID, "mlx_revision": MLX_REVISION, "bits": BITS, "group": GROUP,
        "converted_to": "~/.cache/huggingface/mlcourse/" + path.name, "converted_this_run": converted_now,
        "mlx_copy_vs_hf": copy_check, "ppl_tokens": rows[0]["tokens"], "speed_setup": speed,
        "rows": rows, "safetensors_bytes": files,
        "mlx": {"mlx": mx.__version__, "mlx_lm": __import__("mlx_lm").__version__},
    }
    out["guard"] = guard.report()
    write_results("mlx_way", out)
    if args.figures:
        import figures

        figures.mlx_way()


if __name__ == "__main__":
    main()
