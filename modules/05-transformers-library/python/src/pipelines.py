"""First contact: the three-line experience.

In module 04 you wrote ~300 lines to make attention, a transformer block, and a
sampling loop. Here the *entire* model — tokenizer, weights, generation loop,
sampling — hides behind one call: ``pipeline("text-generation", model=...)``.

Run me:  ``uv run python src/pipelines.py``
"""

from __future__ import annotations

from common import MODEL_ID, get_device


def _device_index() -> int | str:
    dev = get_device()
    # HF pipelines want an index for cuda/cpu, but the string "mps" for Metal.
    return "mps" if dev.type == "mps" else (0 if dev.type == "cuda" else -1)


def text_generation() -> str:
    """Three lines. That's the whole thing."""
    from transformers import pipeline

    gen = pipeline("text-generation", model=MODEL_ID, device=_device_index())
    messages = [{"role": "user", "content": "Give me one surprising fact about octopuses."}]
    out = gen(messages, max_new_tokens=64, do_sample=True, temperature=0.7)
    reply = out[0]["generated_text"][-1]["content"]
    print("\n=== text-generation ===")
    print(reply)
    return reply


def sentiment() -> list[dict]:
    """A different task, the same three-line shape — a small task-specific head."""
    from transformers import pipeline

    clf = pipeline("sentiment-analysis", device=_device_index())
    texts = [
        "I built a transformer from scratch and it finally works!",
        "The download failed for the third time and I am losing my mind.",
    ]
    out = clf(texts)
    print("\n=== sentiment-analysis ===")
    for t, r in zip(texts, out):
        print(f"  [{r['label']:>8} {r['score']:.2f}]  {t}")
    return out


def zero_shot() -> dict:
    """Zero-shot classification: label text with categories the model never trained on."""
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", device=_device_index())
    text = "The new GPU cut our training time from three days to eight hours."
    labels = ["hardware", "cooking", "finance", "sports"]
    out = clf(text, candidate_labels=labels)
    print("\n=== zero-shot-classification ===")
    print(f"  text:   {text}")
    for lbl, score in zip(out["labels"], out["scores"]):
        print(f"    {lbl:>10}: {score:.2f}")
    return out


def main() -> None:
    print(f"device: {get_device()}")
    text_generation()
    sentiment()
    zero_shot()
    print(
        "\nThree tasks, three near-identical calls. In module 04 the "
        "text-generation loop alone was ~300 lines. That is the pipeline."
    )


if __name__ == "__main__":
    main()
