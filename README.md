# Build ML From Scratch — Then the Hugging Face Way

Learn to **build** machine-learning models, not just call them — from a scalar
autograd engine all the way to fine-tuning and serving an LLM. The trick of this
course: you build **everything twice**. First from scratch, so you understand
every moving part; then the Hugging Face way, so you can move fast and ship real
work. You **see every concept as a picture** — loss surfaces, attention
heatmaps, feature maps, denoising trajectories — because intuition comes from
images, not equations alone. And you write the core algorithms in **two
languages**: **Python** for reach and ecosystem, and **Zig** for understanding —
where nothing is hidden, memory is explicit, and you finally see what the tensor
library was doing all along.

This is a hands-on course. There is no "watch me code." Every module is a
walkthrough you run on your own machine, an Apple-silicon Mac, with optional
cloud acceleration when you want it.

---

## How to use this course

Each **module** is a self-contained folder under `modules/` and follows the same
shape:

- **README walkthrough** — the narrative. Read it top to bottom; it explains the
  concept, the math you actually need, and what you are about to build.
- **Runnable scripts** — every code block is a real script you launch with
  [uv](https://docs.astral.sh/uv/). No hidden setup: `uv run script.py` just
  works.
- **Exercises with solutions** — you write code; a reference solution lives in
  `solutions/` so you can check yourself (peek only after you try).
- **Figures** — generated plots and diagrams in `figures/`. Regenerate them
  yourself; seeing the picture is the point.
- **A checkpoint** — each module ends with a short "you should now be able
  to…" checklist. Don't move on until you can tick every box.

**Do the modules in order.** Later modules assume the tools, vocabulary, and
code you built earlier. Foundations (01–03) are load-bearing for everything that
follows.

Before module 01, work through **[SETUP.md](SETUP.md)** to verify your
toolchain. If your Mac ever feels too small — a big fine-tune, a hungry GPU
job — **[CLOUD.md](CLOUD.md)** shows the optional cloud lane.

Prefer a cohesive reader with search, local progress, knowledge checks, and
installable offline support? Build the course app—or prepare a reader/core/full
pack—using **[OFFLINE.md](OFFLINE.md)**.

---

## The learning path

```mermaid
flowchart TD
    subgraph F["Foundations · 01–03"]
        M01["01 · autograd"] --> M02["02 · neural networks"] --> M03["03 · tokenization"]
    end
    subgraph T["Transformers & LLMs · 04–07"]
        M04["04 · attention & transformer"] --> M05["05 · transformers library"] --> M05A["05½ · data & evaluation"] --> M06["06 · fine-tuning"] --> M07["07 · inference internals"]
    end
    subgraph B["Breadth · 08–10"]
        M08["08 · vision"]
        M09["09 · diffusion"]
        M10["10 · agents"]
    end
    F --> T --> B
    M08 -.-> M09 -.-> M10

    classDef track fill:#eef,stroke:#88a,color:#113;
    class F,T,B track;
```

The three tracks build on each other. **Foundations** gives you autograd, a
network, and a tokenizer written by hand. **Transformers & LLMs** turns those
into attention, a real model library, fine-tuning, and fast inference.
**Breadth** takes the same ideas sideways into vision, diffusion, and agents.

---

## Modules

| # | Module | What you build |
|---|--------|----------------|
| 01 | [autograd](modules/01-autograd) | A scalar autograd engine, in **Python and Zig** — reverse-mode backprop from first principles. |
| 02 | [neural-networks](modules/02-neural-networks) | A numpy MLP trained on **FashionMNIST**, loaded via Hugging Face `datasets`. |
| 03 | [tokenization](modules/03-tokenization) | **BPE from scratch** in Python and Zig, then the real **SmolLM3** tokenizer, with a **Gradio** playground. |
| 04 | [attention-transformer](modules/04-attention-transformer) | Attention from scratch → a tiny **char-GPT**, with **attention heatmaps**. |
| 05 | [transformers-library](modules/05-transformers-library) | `pipeline`, `AutoModel`, **SmolLM3 anatomy**, and sampling strategies. |
| 05½ | [data-evaluation](modules/05a-data-evaluation) | Splits, confusion matrices, thresholds, calibration, slices, and leakage checks **from scratch**. |
| 06 | [fine-tuning](modules/06-fine-tuning) | **TRL SFT + LoRA** on SmolLM3, an **MLX** Mac lane, and **trackio** tracking. |
| 07 | [inference-internals](modules/07-inference-internals) | **KV cache**, quantization, and a **Zig inference capstone**. |
| 08 | [vision](modules/08-vision) | Fine-tune a **ViT**, and visualize its **feature maps**. |
| 09 | [diffusion](modules/09-diffusion) | A toy denoiser from scratch → the **`diffusers`** library. |
| 10 | [agents](modules/10-agents) | Build agents with **smolagents** and the **Model Context Protocol (MCP)**. |

---

## Explorables

Every module and the evaluation bridge ship an **interactive, in-browser
explainer** — a single HTML file with no dependencies, no build step and no
network calls. Where a module
produced real artifacts (trained weights, merge lists, attention matrices,
measured timings), its explorable inlines them, so what you drag and scrub is
the same data your own runs produce.

Serve the repo root and open the gallery:

```bash
python3 -m http.server        # then browse to
# http://localhost:8000/explorables/
```

| Explorable | Module | What you do |
|---|---|---|
| [Gradient descent & backprop](explorables/01-gradient-descent.html) | 01 | Roll a ball down a non-convex loss surface, then step a real computation graph forward and backward. |
| [Neural-net playground](explorables/02-nn-playground.html) | 02 | Train the from-scratch net **in your browser** on two moons and watch the decision boundary bend. |
| [BPE, one merge at a time](explorables/03-bpe-stepper.html) | 03 | Step through the real first 60 merges, then apply them to text you type. |
| [Attention, one query at a time](explorables/04-attention.html) | 04 | Drive a Q·Kᵀ heatmap with causal mask and temperature — toy vectors or the real tiny-GPT heads. |
| [Anatomy of a transformer](explorables/05-transformer-anatomy.html) | 05 | Send a pulse up the decoder stack, hover components for real shapes, read a **logit lens**. |
| [The honest scorecard](explorables/05a-evaluation-lab.html) | 05½ | Move a threshold, inspect every error, compare slices, and introduce leakage to manufacture a fake improvement. |
| [Low rank: what LoRA actually trains](explorables/06-lora-rank.html) | 06 | Drag the rank slider and watch a real trained **ΔW** rebuild from 7 singular directions. |
| [KV cache & int8](explorables/07-kv-cache.html) | 07 | Run the decode loop with the cache on and off, every attention cell counted, then quantize. |
| [Convolution vs patches](explorables/08-conv-vs-patches.html) | 08 | Slide real kernels over a real image, then watch it become a **ViT** patch sequence. |
| [Diffusion on a spiral](explorables/09-diffusion.html) | 09 | Scrub a spiral into pure noise, then run the trained denoiser live to pull it back out. |
| [The ReAct loop](explorables/10-agent-loop.html) | 10 | Play a real agent transcript message by message — including the run where it fails. |

### Quizzes

Each of the ten numbered modules — and the 05½ bridge lab — ships a
**10-question quiz**, every question grounded in that module's own code, figures
and measured numbers, with an explanation revealed as soon as you answer. Start at
[`quizzes/index.html`](quizzes/index.html), which lists all eleven and shows your
score for each. Nothing is uploaded: answers and scores stay in your browser's
`localStorage`, and every quiz has a Reset button.

### Course map

The **course map** is a visual dashboard of the whole course on a single page:
one card per module with its signature figure, the measured results that module
actually produced, and direct links to its explorable and quiz. Progress
checkboxes let you tick modules off as you finish them; like the quizzes, they
are stored in your browser and uploaded nowhere.

Read it live at **<https://wynch.github.io/ml-course/map.html>**, or open
`map.html` when serving the repo root locally.

Regenerate it after changing a figure or a number:

```bash
uv run tools/coursemap/build_map.py
```

The source of truth is `tools/coursemap/template.html` — edit that, never the
generated `map.html`, which is the template with every module figure inlined as
a data: URI. The map first lived as a one-off claude.ai artifact; the copy
published here on GitHub Pages is now the maintained home.

---

## Toolchain

The course targets an **Apple-silicon Mac** and a small, modern toolset:

- **[uv](https://docs.astral.sh/uv/)** — Python environment and script runner (one per module).
- **[Zig](https://ziglang.org/)** — the from-scratch systems lane.
- **[`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)** — Hub auth, downloads, and cloud Jobs.
- Optional per-module extras: **PyTorch**, **mlx-lm**, **Gradio**, **graphviz**.

Full verification steps and per-module setup live in **[SETUP.md](SETUP.md)**.
The optional cloud lane — HF Jobs, trackio, Spaces, ZeroGPU — is in
**[CLOUD.md](CLOUD.md)**.

The multi-language philosophy has its own page: **[docs/algorithm-cards.md](docs/algorithm-cards.md)**
explains the "algorithm card" format — the same algorithm side by side in
Python and Zig — and lists the cards this course ships.

---

## Built on

This course stands on the shoulders of the free, open
[Hugging Face learning platform](https://huggingface.co/learn). Each track draws
directly from one or more of these courses — go deeper there any time. For the
full catalog, a module-by-module map, certifications, and suggested continuation
paths — plus a second part covering the best verified free resources *beyond*
Hugging Face, from interactive explainers to university courses and annotated
implementations — see **[RESOURCES.md](RESOURCES.md)**.

- **[LLM Course](https://huggingface.co/learn/llm-course)** — transformers, tokenizers, `datasets`, and fine-tuning.
- **[smol-course](https://huggingface.co/learn/smol-course)** — SmolLM3, TRL, LoRA, and small-model fine-tuning.
- **[Agents Course](https://huggingface.co/learn/agents-course)** — smolagents and agentic patterns.
- **[MCP Course](https://huggingface.co/learn/mcp-course)** — the Model Context Protocol.
- **[Community Computer Vision Course](https://huggingface.co/learn/computer-vision-course)** — ViT and vision transfer learning.
- **[Diffusion Course](https://huggingface.co/learn/diffusion-course)** — denoising diffusion and `diffusers`.
- **[Deep RL Course](https://huggingface.co/learn/deep-rl-course)** — reinforcement learning foundations.
- **[Open-Source AI Cookbook](https://huggingface.co/learn/cookbook)** — practical, task-focused recipes.
