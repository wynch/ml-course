"""Solution (c) — measure and plot SmolLM3 fertility for a chosen language.

Two measures. Tokens per word on the sentences below, two of which are
pangrams: they pile up rare words, so they overstate the gap. And the
"premium" of Petrov et al. (arXiv 2305.15425): total tokens for the same
content, on ordinary sentences and their translations. Word counts differ
between languages, so the premium is the fairer number. `__main__` writes
both to ../results/fertility.json.
"""

from __future__ import annotations

import json
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


# Ordinary sentences and their Spanish translations, for the premium.
PARALLEL = [
    ("The train to the city leaves at eight in the morning.",
     "El tren a la ciudad sale a las ocho de la mañana."),
    ("My sister works in a small hospital near the coast.",
     "Mi hermana trabaja en un pequeño hospital cerca de la costa."),
    ("We need to buy bread, milk and eggs for the weekend.",
     "Tenemos que comprar pan, leche y huevos para el fin de semana."),
    ("The children played in the park until it started to rain.",
     "Los niños jugaron en el parque hasta que empezó a llover."),
    ("Please send me the report before the meeting tomorrow.",
     "Por favor, envíame el informe antes de la reunión de mañana."),
]


def premium(tokenizer, pairs) -> tuple[int, int]:
    """Total tokens for the English side and the translated side."""
    en = sum(len(tokenizer.tokenize(a)) for a, _ in pairs)
    other = sum(len(tokenizer.tokenize(b)) for _, b in pairs)
    return en, other


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
    from transformers import AutoTokenizer

    res = run(save=True)
    for lang, f in res.items():
        print(f"{lang:9s}: {f:.2f} tokens/word")
    tax = res[MY_LANGUAGE] / res["english"]
    print(f"{MY_LANGUAGE} tax vs english (tokens/word, pangrams): {tax:.2f}x")

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    en_tokens, my_tokens = premium(tok, PARALLEL)
    print(f"premium on {len(PARALLEL)} ordinary parallel sentences: "
          f"{en_tokens} vs {my_tokens} tokens = {my_tokens / en_tokens:.2f}x")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "fertility.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"tokenizer": "HuggingFaceTB/SmolLM3-3B",
                   "tokens_per_word": res, "tax_tokens_per_word": round(tax, 4),
                   "parallel_pairs": len(PARALLEL),
                   "parallel_tokens": {"english": en_tokens, MY_LANGUAGE: my_tokens},
                   "premium": round(my_tokens / en_tokens, 4)}, f, indent=1)
        f.write("\n")
    print(f"wrote {out}")
