"""Solution (c) — measure and plot SmolLM3 fertility for a chosen language."""

from __future__ import annotations

import os


def measure_fertility(tokenizer, sentences: list[str]) -> float:
    total_tokens = 0
    total_words = 0
    for s in sentences:
        total_tokens += len(tokenizer.tokenize(s))
        total_words += len(s.split())
    return total_tokens / max(total_words, 1)


MY_LANGUAGE = "spanish"
MY_SENTENCES = [
    "El veloz murciélago hindú comía feliz cardillo y kiwi.",
    "La cigüeña tocaba el saxofón detrás del palenque de paja.",
    "Añoro las mañanas en las que jugábamos al ajedrez junto al río.",
]

ENGLISH_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "I miss the mornings when we played chess by the river.",
]


def plot(fertilities: dict, out_path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    langs = list(fertilities.keys())
    vals = [fertilities[k] for k in langs]
    ax.bar(langs, vals, color=["#4C72B0", "#C44E52"], edgecolor="white")
    ax.set_ylabel("tokens per word (SmolLM3)")
    ax.set_title("SmolLM3 fertility: English vs " + MY_LANGUAGE)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def run(save: bool = False) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    result = {
        "english": measure_fertility(tok, ENGLISH_SENTENCES),
        MY_LANGUAGE: measure_fertility(tok, MY_SENTENCES),
    }
    if save:
        here = os.path.dirname(__file__)
        out = os.path.join(here, "..", "figures", "exercise_c_fertility.png")
        plot(result, out)
    return result


if __name__ == "__main__":
    res = run(save=True)
    for lang, f in res.items():
        print(f"{lang:9s}: {f:.2f} tokens/word")
    print(f"{MY_LANGUAGE} tax vs english: {res[MY_LANGUAGE] / res['english']:.2f}x")
