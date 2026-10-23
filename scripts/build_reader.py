#!/usr/bin/env python3
"""Build the dependency-free course reader and a Sites-compatible artifact."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
READER = ROOT / "reader"

# GitHub Pages serves .py, .zig and .lock as application/octet-stream and .toml
# and .md as types no browser renders, so every one of them downloads instead of
# opening. Each of these gets a sibling <name>.html that displays it. The rule is
# purely the extension, so app.js can apply it without a lookup table.
VIEWABLE = {
    ".cfg", ".csv", ".ini", ".json", ".lock", ".md", ".py", ".sh", ".toml",
    ".txt", ".yaml", ".yml", ".zig", ".zon",
}
# Dotfiles carry no suffix, so only the directory index links their views.
VIEWABLE_NAMES = {".gitignore", ".python-version"}
# Above this, a corpus or a lockfile is something to download, not to read, so
# the view becomes a stub pointing at the raw file. Every viewable path still
# has a view, which is what keeps the extension rule total.
VIEW_MAX_BYTES = 128_000

LANGUAGES = {
    ".py": "Python", ".zig": "Zig", ".zon": "Zig object notation",
    ".toml": "TOML", ".lock": "Lockfile", ".md": "Markdown", ".json": "JSON",
    ".csv": "CSV", ".txt": "Plain text", ".sh": "Shell", ".yml": "YAML",
    ".yaml": "YAML", ".cfg": "Config", ".ini": "Config",
}


def is_viewable(path: Path) -> bool:
    return path.suffix in VIEWABLE or path.name in VIEWABLE_NAMES

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

# Reading order, not number order: the lesson's Previous and Next buttons
# follow this list (each track in turn; 22 comes before 21, and modules 26-33
# sit inside their tracks).
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
    ("33-text-by-denoising", "Text by denoising", "Text by denoising", "modules/33-text-by-denoising/README.md", "Breadth", 75, "33-text-by-denoising.html", "green"),
    ("11-inside-a-language-model", "Inside a language model", "Inside an LM", "modules/11-inside-a-language-model/README.md", "Looking inside", 150, "11-inside-a-language-model.html", "rose"),
    ("12-features-and-dictionaries", "Features and dictionaries", "Features", "modules/12-features-and-dictionaries/README.md", "Looking inside", 150, "12-features-and-dictionaries.html", "rose"),
    ("13-inside-vision-and-jepa", "Inside vision and JEPA models", "Inside vision", "modules/13-inside-vision-and-jepa/README.md", "Looking inside", 90, "13-inside-vision-and-jepa.html", "rose"),
    ("32-is-the-explanation-real", "Is the explanation real?", "Is it real?", "modules/32-is-the-explanation-real/README.md", "Looking inside", 75, "32-is-the-explanation-real.html", "rose"),
    ("14-how-training-finds-a-solution", "How training finds a solution", "Training dynamics", "modules/14-how-training-finds-a-solution/README.md", "How learning works", 90, "14-how-training-finds-a-solution.html", "blue"),
    ("15-geometry-of-representations", "Geometry of representations", "Geometry", "modules/15-geometry-of-representations/README.md", "How learning works", 75, "15-geometry-of-representations.html", "blue"),
    ("16-self-supervised-and-jepa", "Self-supervised learning and JEPA", "SSL and JEPA", "modules/16-self-supervised-and-jepa/README.md", "How learning works", 90, "16-self-supervised-and-jepa.html", "blue"),
    ("29-learning-new-keeping-old", "Learning something new without losing the old", "Learning without losing", "modules/29-learning-new-keeping-old/README.md", "How learning works", 90, "29-learning-new-keeping-old.html", "blue"),
    ("30-learning-without-a-backward-pass", "Learning without a backward pass", "Without a backward pass", "modules/30-learning-without-a-backward-pass/README.md", "How learning works", 80, "30-learning-without-a-backward-pass.html", "blue"),
    ("17-learning-from-preferences", "Learning from preferences", "Preferences", "modules/17-learning-from-preferences/README.md", "Other recipes", 90, "17-learning-from-preferences.html", "rust"),
    ("18-beyond-attention", "Beyond attention", "Recurrence and SSMs", "modules/18-beyond-attention/README.md", "Other recipes", 90, "18-beyond-attention.html", "rust"),
    ("27-memory-as-a-matrix", "Memory as a matrix: the delta rule", "Memory as a matrix", "modules/27-memory-as-a-matrix/README.md", "Other recipes", 75, "27-memory-as-a-matrix.html", "rust"),
    ("19-scaling-laws", "Scaling laws", "Scaling laws", "modules/19-scaling-laws/README.md", "Why they work", 60, "19-scaling-laws.html", "olive"),
    ("20-in-context-learning", "In-context learning", "In-context learning", "modules/20-in-context-learning/README.md", "Why they work", 60, "20-in-context-learning.html", "olive"),
    ("22-reasoning-at-test-time", "Reasoning at test time", "Reasoning at test time", "modules/22-reasoning-at-test-time/README.md", "Using models well", 75, "22-reasoning-at-test-time.html", "ochre"),
    ("21-retrieval-and-memory", "Retrieval and memory", "Retrieval", "modules/21-retrieval-and-memory/README.md", "Using models well", 75, "21-retrieval-and-memory.html", "ochre"),
    ("23-compression", "Compression", "Compression", "modules/23-compression/README.md", "Using models well", 120, "23-compression.html", "ochre"),
    ("28-fewer-bits-trained-in", "Fewer bits, trained in", "Fewer bits", "modules/28-fewer-bits-trained-in/README.md", "Using models well", 120, "28-fewer-bits-trained-in.html", "ochre"),
    ("24-reinforcement-learning", "Reinforcement learning", "RL", "modules/24-reinforcement-learning/README.md", "Acting and planning", 75, "24-reinforcement-learning.html", "slate"),
    ("25-world-models-and-planning", "World models and planning", "World models", "modules/25-world-models-and-planning/README.md", "Acting and planning", 60, "25-world-models-and-planning.html", "slate"),
    ("26-rl-with-a-checker", "RL with a checker", "RL with a checker", "modules/26-rl-with-a-checker/README.md", "Acting and planning", 75, "26-rl-with-a-checker.html", "slate"),
    ("31-world-model-from-pixels", "A world model from pixels", "World model from pixels", "modules/31-world-model-from-pixels/README.md", "Acting and planning", 75, "31-world-model-from-pixels.html", "slate"),
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


MODULE_ROUTES = {module[3].split("/")[1]: module[0] for module in MODULES}
MODULE_LINK = re.compile(r'href="\.\./modules/([^"#]+?)/?(#[^"]*)?"')


def module_target(match: re.Match[str]) -> str:
    """Point one module link in a bundled page at its place in the reader.

    Bundled pages sit exactly one level under the reader root, so the rewritten
    link stays relative (``../``). That keeps the reader working both at a
    domain root and under a subpath such as /ml-course/reader/.
    """
    path, anchor = match.group(1), match.group(2) or ""
    slug, _, rest = path.partition("/")
    route = MODULE_ROUTES.get(slug)
    if route and rest in ("", "README.md"):
        return f'href="../index.html#/{route}"'
    # A bare source path downloads on Pages, so aim at its readable view.
    suffix = ".html" if is_viewable(Path(path)) else ""
    return f'href="../content/modules/{path}{suffix}{anchor}"'


def route_module_links(directory: Path) -> int:
    """Rewrite ../modules/… hrefs in the reader's copy of a bundled directory.

    The bundled quizzes and explorables are written for the repository layout,
    where they sit one level under the module tree. The reader keeps them at its
    root and serves lessons from a hash route, so a link left alone dead-ends.
    """
    rewritten = 0
    for page in sorted(directory.rglob("*.html")):
        updated, count = MODULE_LINK.subn(module_target, page.read_text())
        if count:
            page.write_text(updated)
            rewritten += count
    return rewritten


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
                "explorable": f"explorables/{explorable}",
                "accent": accent,
                "markdown": (ROOT / path).read_text(),
            }
        )
    return {"guides": guides, "modules": modules, "quizzes": QUIZZES}


# What the service worker saves on install. A phone has little room, so the
# list keeps what a reader needs offline and leaves the rest online-only: every
# file stays in the site and its links still work, it is just not pre-cached.
OFFLINE_MAX_BYTES = 2_000_000
# Model weights, optimizer states and data blobs: never read, only downloaded.
BLOB_SUFFIXES = {
    ".bin", ".ckpt", ".gguf", ".h5", ".npy", ".npz", ".onnx", ".pkl",
    ".pickle", ".pt", ".pth", ".safetensors",
}
# Warm order. Explorables, quizzes and research come straight after the shell,
# so a phone that runs out of room or cuts the install short loses the long
# tail of source files, never the visualizations.
OFFLINE_GROUPS = ("core", "explorables", "quizzes", "research", "lessons", "rest")
# Reader paths under content/ map back to these source folders and files.
CONTENT_SOURCES = ("modules", "docs", "offline")
MARKDOWN_IMAGE = re.compile(r'!\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
PAGE_REFERENCE = re.compile(r'(?:href|src)="([^"#?]+)')


def tracked_files() -> set[str] | None:
    """The course files git tracks, relative to the course root.

    A local checkout holds checkpoints and runs that CI never sees; listing only
    tracked files keeps a local build's offline list identical to the published
    one. Outside a git checkout every file counts.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {path for path in result.stdout.decode().split("\0") if path}


def source_of(asset: str) -> str | None:
    """The tracked source path a copied reader file came from, if any."""
    parts = asset.split("/")
    if parts[0] in ("explorables", "quizzes", "research"):
        return asset
    if parts[0] == "content":
        rest = "/".join(parts[1:])
        if len(parts) == 2 or parts[1] in CONTENT_SOURCES:
            return rest
    return None


def referenced_assets(root: Path, payload: dict) -> set[str]:
    """Files a lesson shows or a bundled page loads, which stay in at any size."""
    found: set[str] = set()
    for page in payload["guides"] + payload["modules"]:
        base = "/".join(page["path"].split("/")[:-1])
        for source in MARKDOWN_IMAGE.findall(page["markdown"]):
            if not re.match(r"^(https?:|data:|/)", source):
                found.add(f"content/{normalize(f'{base}/{source}')}")
    for folder in ("explorables", "quizzes", "research"):
        for html in (root / folder).rglob("*.html"):
            directory = html.parent.relative_to(root).as_posix()
            for href in PAGE_REFERENCE.findall(html.read_text()):
                if not re.match(r"^(https?:|mailto:|data:|//)", href):
                    found.add(normalize(f"{directory}/{href}"))
    return found


def normalize(path: str) -> str:
    parts: list[str] = []
    for part in path.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def offline_reason(asset: str, root: Path, tracked: set[str] | None,
                   tracked_dirs: set[str], views: set[str], indexes: set[str],
                   referenced: set[str]) -> str | None:
    """Why a reader file stays online-only, or None when it is saved offline."""
    path = Path(asset)
    # A source view or a folder page is judged by what it shows.
    subject = asset[: -len(".html")] if asset in views else asset
    if any(part.startswith(".") for part in Path(subject).parts[:-1]):
        return "hidden folder"
    name = Path(subject).name
    if Path(subject).suffix in BLOB_SUFFIXES or "optimizer" in name.lower():
        return "weights or data blob"
    if Path(subject).suffix == ".lock":
        return "lockfile"
    if tracked is not None:
        if asset in indexes:
            folder = source_of(str(path.parent)) if path.parent.as_posix() != "content" else ""
            if folder is not None and folder and folder not in tracked_dirs:
                return "untracked"
        else:
            source = source_of(subject)
            if source is not None and source not in tracked:
                return "untracked"
    size = (root / asset).stat().st_size
    if asset in referenced:
        return None
    if size > OFFLINE_MAX_BYTES:
        return "too large"
    # Past this size the reader shows a stub instead of the file, so the raw
    # corpus or result dump is only ever downloaded, never read in place.
    if asset not in views and is_viewable(path) and size > VIEW_MAX_BYTES:
        return "raw file behind a stub view"
    return None


def offline_group(asset: str, lesson_files: set[str]) -> str:
    top = asset.split("/")[0]
    if "/" not in asset:
        return "core"
    if top in ("explorables", "quizzes", "research"):
        return top
    if asset in lesson_files:
        return "lessons"
    return "rest"


def offline_manifest(root: Path, payload: dict, views: set[Path]) -> dict:
    """The priority-ordered list of what the service worker saves on install.

    Every URL is relative to the reader root; the service worker resolves them
    against its own script URL, so one build serves from a domain root or a
    subpath.
    """
    tracked = tracked_files()
    tracked_dirs = {
        "/".join(path.split("/")[:end])
        for path in (tracked or ())
        for end in range(1, path.count("/") + 1)
    }
    view_paths = {view.relative_to(root).as_posix() for view in views}
    index_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "content").rglob("index.html")
        if path.relative_to(root).as_posix() not in view_paths
    }
    referenced = referenced_assets(root, payload)

    lessons: dict[str, list[str]] = {}
    lesson_files: set[str] = set()
    for module in payload["modules"]:
        readme = f"content/{module['path']}"
        base = "/".join(module["path"].split("/")[:-1])
        figures = sorted({
            f"content/{normalize(f'{base}/{source}')}"
            for source in MARKDOWN_IMAGE.findall(module["markdown"])
            if not re.match(r"^(https?:|data:|/)", source)
        })
        lessons[module["id"]] = figures
        lesson_files.update([readme, f"{readme}.html", *figures])

    groups: dict[str, list[str]] = {group: [] for group in OFFLINE_GROUPS}
    skipped: dict[str, int] = {}
    skipped_bytes = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "offline-assets.json":
            continue
        asset = path.relative_to(root).as_posix()
        reason = offline_reason(
            asset, root, tracked, tracked_dirs, view_paths, index_paths, referenced
        )
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            skipped_bytes += path.stat().st_size
            continue
        total_bytes += path.stat().st_size
        groups[offline_group(asset, lesson_files)].append(asset)

    included = {asset for urls in groups.values() for asset in urls}
    # A lesson counts as saved when its text (in course-content.js) and every
    # figure it shows are cached, so only figures that are saved are listed.
    lessons = {
        module: [figure for figure in figures if figure in included]
        for module, figures in lessons.items()
    }
    # Within a group, pages first, so a half-finished group still opens.
    for group in ("explorables", "quizzes", "research"):
        groups[group].sort(key=lambda asset: (not asset.endswith(".html"), asset))
    groups["lessons"].sort(key=lambda asset: (not asset.endswith("README.md.html"), asset))
    return {
        "version": 2,
        "bytes": total_bytes,
        "groups": [{"id": group, "urls": groups[group]} for group in OFFLINE_GROUPS],
        "lessons": lessons,
        "skipped": dict(sorted(skipped.items())),
        "skippedBytes": skipped_bytes,
    }


BROWSE_CSS = """/* Generated by scripts/build_reader.py for the bundled source tree. */
:root {
  --paper: #f4f0e7; --panel: #fffdf8; --ink: #1c2422; --muted: #61706b;
  --line: rgba(28, 36, 34, 0.15); --teal: #1f918d; --code: #19211f;
  --code-ink: #e9f1ed;
  --body: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #111714; --panel: #19221f; --ink: #e8efec; --muted: #a1aea9;
    --line: rgba(232, 239, 236, 0.17); --teal: #3ab6b0; --code: #090d0c;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.5rem 4rem; background: var(--paper);
  color: var(--ink); font-family: var(--body); line-height: 1.5;
}
main { max-width: 60rem; margin: 0 auto; }
a { color: var(--teal); }
.back {
  display: inline-block; margin-bottom: 1.5rem; font-family: var(--mono);
  font-size: .8rem; text-decoration: none;
}
.back:hover { text-decoration: underline; }
h1 {
  font-family: var(--mono); font-size: 1.15rem; font-weight: 600;
  margin: 0 0 .35rem; word-break: break-all;
}
.crumb {
  font-family: var(--mono); font-size: .78rem; color: var(--muted);
  margin: 0 0 1.75rem; word-break: break-all;
}
.listing { border-top: 1px solid var(--line); }
.listing a {
  display: flex; gap: 1rem; align-items: baseline; padding: .55rem .4rem;
  border-bottom: 1px solid var(--line); text-decoration: none; color: var(--ink);
}
.listing a:hover { background: var(--panel); color: var(--teal); }
.listing .name { font-family: var(--mono); font-size: .85rem; flex: 1; word-break: break-all; }
.listing .size {
  font-family: var(--mono); font-size: .74rem; color: var(--muted);
  white-space: nowrap;
}
.listing .dir { color: var(--teal); }
.empty { color: var(--muted); font-size: .9rem; }
.source {
  display: flex; background: var(--code); color: var(--code-ink);
  border-radius: 8px; overflow: auto; font-family: var(--mono);
  font-size: .82rem; line-height: 1.6;
}
/* Only the block padding here: the columns below set their own inline
   padding, and a shorthand would outspecify them. */
.source pre { margin: 0; padding-block: 1rem; }
.gutter {
  padding-inline: 1rem .85rem; text-align: right;
  color: var(--code-ink); opacity: .35; user-select: none;
  border-right: 1px solid rgba(233, 241, 237, .12);
}
.code { padding-inline: 1rem 1.5rem; flex: 1; }
.stub {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 1.25rem 1.4rem; font-size: .92rem;
}
"""


# Kept at the course root so the site pages share them; copied to the reader root.
ICONS = ("favicon.svg", "favicon.ico", "apple-touch-icon.png")


def human_size(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.1f} MB"


def up_to_root(directory: Path) -> str:
    """The relative hop from a directory in the reader back to its root."""
    return "../" * len(directory.relative_to(READER).parts)


def page(directory: Path, title: str, crumb: str, back: str, body: str) -> str:
    up = up_to_root(directory)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{escape(title)}</title>\n"
        f"<link rel=\"icon\" href=\"{up}favicon.svg\" type=\"image/svg+xml\">\n"
        f"<link rel=\"icon\" href=\"{up}favicon.ico\" sizes=\"any\">\n"
        f"<link rel=\"apple-touch-icon\" href=\"{up}apple-touch-icon.png\">\n"
        f"<link rel=\"stylesheet\" href=\"{up}browse.css\">\n</head>\n<body>\n<main>\n"
        f"{back}\n<h1>{escape(title)}</h1>\n<p class=\"crumb\">{escape(crumb)}</p>\n"
        f"{body}\n</main>\n</body>\n</html>\n"
    )


def source_view(path: Path) -> str:
    """One readable page for a file the browser would otherwise download."""
    label = LANGUAGES.get(path.suffix, "Source file")
    size = path.stat().st_size
    crumb = f"{label} · {human_size(size)} · {path.relative_to(READER / 'content')}"
    back = '<a class="back" href="index.html">← back to this folder</a>'
    if size > VIEW_MAX_BYTES:
        body = (
            f'<div class="stub"><p>This file is {human_size(size)} — too large to '
            "read comfortably in a browser, so it is not inlined here.</p>"
            f'<p><a href="{escape(path.name)}">Download the raw file</a></p></div>'
        )
        return page(path.parent, path.name, crumb, back, body)

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    numbers = "\n".join(str(number) for number in range(1, len(lines) + 1))
    source = escape("\n".join(lines), quote=False)
    body = (
        f'<p class="crumb"><a href="{escape(path.name)}">Open the raw file</a></p>\n'
        '<div class="source">'
        f'<pre class="gutter" aria-hidden="true">{numbers}</pre>'
        f'<pre class="code"><code>{source}</code></pre>'
        "</div>"
    )
    return page(path.parent, path.name, crumb, back, body)


def write_source_views(content: Path) -> set[Path]:
    generated: set[Path] = set()
    for path in sorted(content.rglob("*")):
        if not path.is_file() or not is_viewable(path):
            continue
        view = path.with_name(f"{path.name}.html")
        if view.exists():
            raise RuntimeError(f"a real file already occupies the view path {view}")
        view.write_text(source_view(path))
        generated.add(view)
    return generated


def directory_index(directory: Path, content: Path, listing: dict[Path, list[Path]]) -> str:
    rows = []
    for child in sorted(listing[directory], key=lambda c: (c.is_file(), c.name.lower())):
        if child.is_dir():
            count = len(listing[child])
            rows.append(
                f'<a href="{escape(child.name)}/index.html"><span class="name dir">'
                f'{escape(child.name)}/</span>'
                f'<span class="size">{count} item{"" if count == 1 else "s"}</span></a>'
            )
        else:
            href = f"{child.name}.html" if is_viewable(child) else child.name
            rows.append(
                f'<a href="{escape(href)}"><span class="name">{escape(child.name)}'
                f'</span><span class="size">{human_size(child.stat().st_size)}'
                "</span></a>"
            )
    body = (
        f'<div class="listing">{"".join(rows)}</div>'
        if rows
        else '<p class="empty">This folder is empty.</p>'
    )
    if directory == content:
        back = '<a class="back" href="../index.html">← back to the course reader</a>'
        title = "Course source"
        crumb = "Every module, guide and figure the reader bundles."
    else:
        back = (
            '<a class="back" href="../index.html">← '
            f"{escape(directory.parent.name)}/</a>"
        )
        title = f"{directory.name}/"
        crumb = str(directory.relative_to(content))
    return page(directory, title, crumb, back, body)


def write_directory_indexes(content: Path, generated: set[Path]) -> int:
    """Give every bundled folder a page, so folder links stop 404ing.

    The listing is snapshotted before anything is written, so a folder's own
    index never shows up inside it and the counts do not depend on walk order.
    """
    directories = [content, *(p for p in content.rglob("*") if p.is_dir())]
    listing = {
        directory: [
            child for child in directory.iterdir() if child not in generated
        ]
        for directory in directories
    }
    for directory in directories:
        index = directory / "index.html"
        if index.exists():
            raise RuntimeError(f"a real file already occupies the index path {index}")
        index.write_text(directory_index(directory, content, listing))
    return len(directories)


def main() -> None:
    shutil.rmtree(READER, ignore_errors=True)
    READER.mkdir(parents=True)
    for filename in ("index.html", "styles.css", "app.js", "sw.js", "manifest.webmanifest"):
        shutil.copy2(SITE / filename, READER / filename)
    for filename in ICONS:
        shutil.copy2(ROOT / filename, READER / filename)

    copy_filtered(ROOT / "modules", READER / "content" / "modules")
    copy_filtered(ROOT / "docs", READER / "content" / "docs")
    # OFFLINE.md links the pack manifest, so bundle it with the rest.
    copy_filtered(ROOT / "offline", READER / "content" / "offline")
    copy_filtered(ROOT / "explorables", READER / "explorables")
    copy_filtered(ROOT / "quizzes", READER / "quizzes")
    # The research companion links back with ../index.html, which lands on the
    # reader home here and on the course home on the site.
    copy_filtered(ROOT / "research", READER / "research")
    routed = route_module_links(READER / "explorables") + route_module_links(READER / "quizzes")
    for filename in ("README.md", "SETUP.md", "OFFLINE.md", "RESOURCES.md", "CLOUD.md"):
        shutil.copy2(ROOT / filename, READER / "content" / filename)
    if (ROOT / "public" / "og.png").exists():
        shutil.copy2(ROOT / "public" / "og.png", READER / "og.png")

    # Both passes run before the asset list is taken, so everything they write
    # is cached by the service worker and available offline.
    (READER / "browse.css").write_text(BROWSE_CSS)
    views = write_source_views(READER / "content")
    indexes = write_directory_indexes(READER / "content", views)

    data = page_payload()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (READER / "course-content.js").write_text(f"window.ML_COURSE={payload};\n")
    manifest = offline_manifest(READER, data, views)
    (READER / "offline-assets.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    entries = sum(len(group["urls"]) for group in manifest["groups"])
    counts = ", ".join(
        f"{group['id']} {len(group['urls'])}" for group in manifest["groups"]
    )

    print(
        f"Built reader/ with {entries} offline assets "
        f"({manifest['bytes'] / 1e6:.1f} MB: {counts}); "
        f"{sum(manifest['skipped'].values())} files online-only "
        f"({manifest['skippedBytes'] / 1e6:.1f} MB: {manifest['skipped']}); "
        f"{routed} module links routed in bundled pages, "
        f"{len(views)} source views and {indexes} folder indexes."
    )

if __name__ == "__main__":
    main()
