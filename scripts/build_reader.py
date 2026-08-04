#!/usr/bin/env python3
"""Build the dependency-free course reader and a Sites-compatible artifact."""

from __future__ import annotations

import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
READER = ROOT / "reader"

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".zig-cache",
    "__pycache__",
    "adapters",
    "data",
    "models",
    "node_modules",
    "outputs",
    "zig-out",
}

GUIDES = [
    ("course", "Course overview", "Overview", "README.md"),
    ("setup", "Set up your machine", "Setup", "SETUP.md"),
    ("offline", "Use the course offline", "Offline packs", "OFFLINE.md"),
    (
        "algorithm-cards",
        "Python ↔ Zig algorithm cards",
        "Algorithm cards",
        "docs/algorithm-cards.md",
    ),
    ("resources", "Resources and continuations", "Resources", "RESOURCES.md"),
    ("cloud", "Optional cloud lane", "Cloud lane", "CLOUD.md"),
]

MODULES = [
    ("00a-perceptron", "The perceptron & least squares", "Perceptron", "modules/00a-perceptron/README.md", "Origins", 120, "00a-perceptron.html", "amber"),
    ("00b-bayes-knn-pca", "Probability, neighbours & eigenvectors", "Bayes, k-NN & PCA", "modules/00b-bayes-knn-pca/README.md", "Origins", 120, "00b-pca.html", "amber"),
    ("00c-kernels-hopfield", "Kernels, memory & the modern bridge", "Kernels & Hopfield", "modules/00c-kernels-hopfield/README.md", "Origins", 150, "00c-double-descent.html", "amber"),
    ("01-autograd", "Autograd from scratch", "Autograd", "modules/01-autograd/README.md", "Foundations", 120, "01-gradient-descent.html", "violet"),
    ("02-neural-networks", "Neural networks & the training loop", "Neural networks", "modules/02-neural-networks/README.md", "Foundations", 150, "02-nn-playground.html", "violet"),
    ("03-tokenization", "Tokenization: from bytes to tokens", "Tokenization", "modules/03-tokenization/README.md", "Foundations", 120, "03-bpe-stepper.html", "violet"),
    ("04-attention-transformer", "Attention & the transformer", "Attention", "modules/04-attention-transformer/README.md", "Transformers & LLMs", 180, "04-attention.html", "teal"),
    ("05-transformers-library", "The transformers library", "Transformers", "modules/05-transformers-library/README.md", "Transformers & LLMs", 150, "05-transformer-anatomy.html", "teal"),
    ("05a-data-evaluation", "Data & evaluation", "Data & evaluation", "modules/05a-data-evaluation/README.md", "Transformers & LLMs", 90, "05a-evaluation-lab.html", "orange"),
    ("06-fine-tuning", "Fine-tuning: make the model yours", "Fine-tuning", "modules/06-fine-tuning/README.md", "Transformers & LLMs", 210, "06-lora-rank.html", "teal"),
    ("07-inference-internals", "Inference internals", "Inference", "modules/07-inference-internals/README.md", "Transformers & LLMs", 210, "07-kv-cache.html", "teal"),
    ("08-vision", "Vision: convolutions to ViTs", "Vision", "modules/08-vision/README.md", "Breadth", 180, "08-conv-vs-patches.html", "green"),
    ("09-diffusion", "Diffusion: learning to denoise", "Diffusion", "modules/09-diffusion/README.md", "Breadth", 180, "09-diffusion.html", "green"),
    ("10-agents", "Agents: models that act", "Agents", "modules/10-agents/README.md", "Breadth", 180, "10-agent-loop.html", "green"),
]

QUIZZES = {
    "01-autograd": [
        {
            "id": "chain-rule",
            "prompt": "In reverse-mode autodiff, what flows backward along each edge?",
            "choices": [
                "The original input value",
                "A local derivative multiplied by the gradient arriving from downstream",
                "A fresh numerical approximation of the entire function",
            ],
            "correct": 1,
            "explanation": "Each node receives an upstream gradient and multiplies it by its local derivative. Contributions from multiple downstream paths accumulate.",
        },
        {
            "id": "topology",
            "prompt": "Why process the graph in reverse topological order?",
            "choices": [
                "So every node has received all downstream gradient contributions before it propagates",
                "Because Python recursion requires it",
                "To keep the forward values sorted numerically",
            ],
            "correct": 0,
            "explanation": "A parent must wait until every consumer has contributed to its gradient. Reverse topological order guarantees that dependency.",
        },
        {
            "id": "gradcheck",
            "prompt": "A finite-difference gradient check is primarily useful for…",
            "choices": [
                "Making training faster",
                "Finding mistakes in analytic backward rules",
                "Selecting the best hidden-layer width",
            ],
            "correct": 1,
            "explanation": "Finite differences are slow but independent of the backward implementation, which makes them a strong debugging oracle.",
        },
    ],
    "02-neural-networks": [
        {
            "id": "softmax",
            "prompt": "Why is softmax usually computed after subtracting the largest logit?",
            "choices": [
                "It changes the winning class",
                "It prevents exponential overflow without changing the probabilities",
                "It makes every class equally likely",
            ],
            "correct": 1,
            "explanation": "Softmax is invariant to adding or subtracting one constant from every logit. Subtracting the maximum keeps exponentials in a safe range.",
        },
        {
            "id": "batch",
            "prompt": "What does a mini-batch gradient estimate trade?",
            "choices": [
                "More noise for cheaper, more frequent updates",
                "Exactness for a different model architecture",
                "Accuracy for permanently lower memory use at inference",
            ],
            "correct": 0,
            "explanation": "A batch samples the full-data gradient. Smaller batches are noisier but cheaper per update and can be useful regularizers.",
        },
        {
            "id": "nonlinearity",
            "prompt": "What happens if every layer in a multilayer network is linear?",
            "choices": [
                "Depth still creates arbitrary curved boundaries",
                "The composition collapses to one linear transformation",
                "Backpropagation becomes impossible",
            ],
            "correct": 1,
            "explanation": "A composition of linear maps is another linear map. Nonlinear activations are what let depth represent curved decision boundaries.",
        },
    ],
    "03-tokenization": [
        {
            "id": "bpe-merge",
            "prompt": "At each BPE training step, what is added to the vocabulary?",
            "choices": [
                "The rarest character",
                "The most frequent adjacent token pair merged into one token",
                "Every word not seen before",
            ],
            "correct": 1,
            "explanation": "BPE repeatedly counts adjacent pairs, merges the most frequent one, and applies that merge throughout the training corpus.",
        },
        {
            "id": "bytes",
            "prompt": "Why start a tokenizer from bytes rather than a fixed character alphabet?",
            "choices": [
                "Bytes guarantee any input can be represented",
                "Bytes always produce fewer tokens than words",
                "Transformers can only multiply byte values",
            ],
            "correct": 0,
            "explanation": "UTF-8 text is bytes, so a byte-level base vocabulary has no unknown character. Common byte sequences can then be merged.",
        },
        {
            "id": "fertility",
            "prompt": "High tokenizer fertility for a language means…",
            "choices": [
                "The language needs fewer training examples",
                "Its text is split into more tokens per word",
                "Every word receives a unique token",
            ],
            "correct": 1,
            "explanation": "Fertility is tokens per word. Higher fertility uses more context and compute for the same amount of human-readable text.",
        },
    ],
}


def copy_filtered(source: Path, target: Path) -> None:
    if source.is_dir():
        if source.name in EXCLUDED_DIRS:
            return
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            copy_filtered(child, target / child.name)
    elif source.is_file() and source.name != ".DS_Store":
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def page_payload() -> dict:
    guides = []
    for page_id, title, short_title, path in GUIDES:
        guides.append(
            {
                "id": page_id,
                "title": title,
                "shortTitle": short_title,
                "path": path,
                "kind": "guide",
                "markdown": (ROOT / path).read_text(),
            }
        )
    modules = []
    for (
        page_id,
        title,
        short_title,
        path,
        track,
        minutes,
        explorable,
        accent,
    ) in MODULES:
        modules.append(
            {
                "id": page_id,
                "title": title,
                "shortTitle": short_title,
                "path": path,
                "kind": "module",
                "track": track,
                "minutes": minutes,
                "explorable": f"/explorables/{explorable}",
                "accent": accent,
                "markdown": (ROOT / path).read_text(),
            }
        )
    return {"guides": guides, "modules": modules, "quizzes": QUIZZES}


def asset_urls(root: Path) -> list[str]:
    return sorted(
        f"/{path.relative_to(root).as_posix()}"
        for path in root.rglob("*")
        if path.is_file() and path.name != "offline-assets.json"
    )


def main() -> None:
    shutil.rmtree(READER, ignore_errors=True)
    READER.mkdir(parents=True)
    for filename in ("index.html", "styles.css", "app.js", "sw.js", "manifest.webmanifest"):
        shutil.copy2(SITE / filename, READER / filename)

    copy_filtered(ROOT / "modules", READER / "content" / "modules")
    copy_filtered(ROOT / "docs", READER / "content" / "docs")
    copy_filtered(ROOT / "explorables", READER / "explorables")
    copy_filtered(ROOT / "quizzes", READER / "quizzes")
    for filename in ("README.md", "SETUP.md", "OFFLINE.md", "RESOURCES.md", "CLOUD.md"):
        shutil.copy2(ROOT / filename, READER / "content" / filename)
    if (ROOT / "public" / "og.png").exists():
        shutil.copy2(ROOT / "public" / "og.png", READER / "og.png")

    payload = json.dumps(page_payload(), ensure_ascii=False, separators=(",", ":"))
    (READER / "course-content.js").write_text(f"window.ML_COURSE={payload};\n")
    assets = asset_urls(READER)
    (READER / "offline-assets.json").write_text(
        json.dumps(assets, indent=2) + "\n"
    )

    print(f"Built reader/ with {len(assets)} offline assets.")


if __name__ == "__main__":
    main()
