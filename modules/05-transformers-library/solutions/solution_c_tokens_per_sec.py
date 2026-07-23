"""Reference solution — exercise (c), tokens/sec MPS vs CPU."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import torch

from common import MODEL_ID, load_tokenizer

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

PROMPT_LENS = [8, 64, 256]
NEW_TOKENS = 64


def _make_prompt(tokenizer, n_tokens: int) -> str:
    filler = ("The history of computing is long and full of curious machines. " * 40)
    ids = tokenizer(filler, add_special_tokens=False).input_ids[:n_tokens]
    return tokenizer.decode(ids)


def time_generation(model, tokenizer, prompt: str, new_tokens: int) -> float:
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        model.generate(**ids, max_new_tokens=4, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id)

    start = time.perf_counter()
    with torch.no_grad():
        model.generate(**ids, max_new_tokens=new_tokens, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start

    return new_tokens / elapsed


def benchmark() -> dict[str, list[float]]:
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer()
    results: dict[str, list[float]] = {}
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    for dev in devices:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev)
        model.eval()
        speeds = []
        for n in PROMPT_LENS:
            prompt = _make_prompt(tokenizer, n)
            speeds.append(time_generation(model, tokenizer, prompt, NEW_TOKENS))
        results[dev] = speeds
        del model
    return results


def plot(results: dict[str, list[float]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(PROMPT_LENS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"cpu": "#C44E52", "mps": "#55A868"}
    for i, (dev, speeds) in enumerate(results.items()):
        ax.bar(x + i * w, speeds, w, label=dev.upper(), color=colors.get(dev, "#4C72B0"))
        for xi, s in zip(x + i * w, speeds):
            ax.text(xi, s, f"{s:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + w / 2 if len(results) > 1 else x)
    ax.set_xticklabels([f"{n} tok" for n in PROMPT_LENS])
    ax.set_xlabel("prompt length")
    ax.set_ylabel("tokens / sec (greedy decode)")
    ax.set_title(f"SmolLM2-360M decode throughput ({NEW_TOKENS} new tokens)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    results = benchmark()
    for dev, speeds in results.items():
        pretty = ", ".join(f"{n}tok={s:.0f}t/s" for n, s in zip(PROMPT_LENS, speeds))
        print(f"{dev.upper():>4}: {pretty}")
    FIG_DIR.mkdir(exist_ok=True)
    plot(results, FIG_DIR / "tokens_per_sec.png")


if __name__ == "__main__":
    main()
