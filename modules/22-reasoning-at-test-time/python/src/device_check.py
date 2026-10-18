"""One short forward pass on the Apple GPU and on the CPU, compared.

    uv run python src/device_check.py

The first 32 tokens of test problem 0 (raw question, no chat template, no
cache) go through Qwen3-1.7B-4bit on MLX's GPU and on its CPU, twice: as the
module runs it (bfloat16 activations and scales) and cast to float32. The
4-bit weights are the same in both. A one-layer test then compares MLX's
4-bit matrix product against a numpy product that rounds every step to
bfloat16. Writes results/device_check.json.
"""

from __future__ import annotations

import numpy as np

from common import GEN, MODEL_ID, MODEL_REVISION, Guard, read_jsonl, write_results

N_TOKENS = 32


def bf16(a: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bfloat16 (ties to even), kept as float32."""
    b = a.astype(np.float32).view(np.uint32)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.view(np.float32)


def compare(a, b) -> dict:
    d = np.abs(a - b)
    agree = a.argmax(-1) == b.argmax(-1)
    return {"max_abs_diff": round(float(d.max()), 5), "mean_abs_diff": round(float(d.mean()), 6),
            "top1_agree_positions": int(agree.sum()), "same_top1_last_position": bool(agree[-1])}


def main() -> None:
    guard = Guard("device_check")
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])
    model, tok = load(path)
    question = read_jsonl(GEN / "problems.jsonl")[0]["question"]
    ids = tok._tokenizer.encode(question, add_special_tokens=False)[:N_TOKENS]
    x = mx.array([ids])

    def logits(device) -> np.ndarray:
        mx.set_default_device(device)
        y = model(x)[0].astype(mx.float32)
        mx.eval(y)
        return np.array(y)

    run = {"gpu_bf16": logits(mx.gpu), "cpu_bf16": logits(mx.cpu)}
    model.set_dtype(mx.float32)          # scales, biases, norms, activations; 4-bit codes untouched
    mx.eval(model.parameters())
    run["gpu_f32"], run["cpu_f32"] = logits(mx.gpu), logits(mx.cpu)

    # One layer: q_proj of block 0, 16 random rows, bfloat16 in and out.
    q = model.model.layers[0].self_attn.q_proj
    xs = bf16(np.random.default_rng(0).standard_normal((16, q.scales.shape[1] * q.group_size)) * 0.5)
    w = mx.dequantize(q.weight, q.scales.astype(mx.bfloat16), q.biases.astype(mx.bfloat16),
                      group_size=q.group_size, bits=q.bits, stream=mx.cpu)
    W = np.array(w.astype(mx.float32))                      # bfloat16 weights, as the CPU kernel forms them
    exact = xs.astype(np.float64) @ W.astype(np.float64).T
    acc = np.zeros(exact.shape, np.float32)
    for k in range(xs.shape[1]):                            # sum in input order, rounding each step to bfloat16
        acc = bf16(acc + bf16(xs[:, k:k + 1] * W[:, k][None, :]))
    layer = {"bf16_running_sum_numpy": round(float(np.abs(acc - exact).max()), 5)}
    for name, dev in (("gpu", mx.gpu), ("cpu", mx.cpu)):
        mx.set_default_device(dev)
        y = mx.quantized_matmul(mx.array(xs).astype(mx.bfloat16), q.weight, q.scales.astype(mx.bfloat16),
                                q.biases.astype(mx.bfloat16), transpose=True,
                                group_size=q.group_size, bits=q.bits)
        mx.eval(y)
        layer[f"{name}_qmm_bf16"] = round(float(np.abs(np.array(y.astype(mx.float32)) - exact).max()), 5)

    out = {
        "tokens": len(ids),
        "logit_range": [round(float(run["gpu_bf16"].min()), 2), round(float(run["gpu_bf16"].max()), 2)],
        "gpu_vs_cpu_bf16": compare(run["gpu_bf16"], run["cpu_bf16"]),
        "gpu_vs_cpu_f32": compare(run["gpu_f32"], run["cpu_f32"]),
        "gpu_bf16_vs_gpu_f32": compare(run["gpu_bf16"], run["gpu_f32"]),
        "cpu_bf16_vs_cpu_f32": compare(run["cpu_bf16"], run["cpu_f32"]),
        "one_layer_max_error_vs_float64": layer,
    }
    print(out)
    out["guard"] = guard.report()
    write_results("device_check", out)


if __name__ == "__main__":
    main()
