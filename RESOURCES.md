# Resources — where to go next

This course was distilled from the free, open [Hugging Face learning
platform](https://huggingface.co/learn). We took the parts that teach you to
**build** — autograd, attention, tokenizers, fine-tuning — and rebuilt them from
scratch, twice, with pictures. But HF Learn is much bigger than what we cover,
and the open web beyond it is bigger still. Every one of our modules has a
natural sequel in both places.

This page is the full menu, in two parts. **Part 1** maps each of our modules
back to the HF Learn course that goes deeper, then catalogs every course on the
platform — what it covers, what you need first, how hands-on it is, and whether
it grants a certificate. **Part 2** is everything else: the explorables,
from-scratch courses, annotated implementations, university syllabi, and
in-browser demos that live outside Hugging Face. Every link in Part 2 was
fetched and verified in **July 2026**, with cost and staleness stated honestly.

---

## Contents

**Part 1 · [Hugging Face Learn](#part-1--hugging-face-learn)**

- [Go deeper, module by module](#go-deeper-module-by-module)
- [The full catalog](#the-full-catalog)
- [Certifications you can collect](#certifications-you-can-collect)
- [Suggested continuation paths](#suggested-continuation-paths)

**Part 2 · [Beyond Hugging Face](#part-2--beyond-hugging-face)**

- [The five must-reads](#the-five-must-reads)
- [Further study, module by module](#further-study-module-by-module)
- [Interactive and visual explainers](#interactive-and-visual-explainers)
- [From-scratch courses and series](#from-scratch-courses-and-series)
- [RL and agents](#rl-and-agents)
- [Math foundations](#math-foundations)
- [Papers as pedagogy](#papers-as-pedagogy)
- [Graphical-first and in-browser](#graphical-first-and-in-browser)
- [Link rot notes](#link-rot-notes)

---

# Part 1 · Hugging Face Learn

## Go deeper, module by module

Each row: what you built here → where to continue on HF Learn.

| Our module | Continue with | Why |
|---|---|---|
| [01 · autograd](modules/01-autograd) · [02 · neural-networks](modules/02-neural-networks) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 1–2 · [CV Course](https://huggingface.co/learn/computer-vision-course) unit 1 | Foundations background: what a Transformer is, and the fundamentals-of-vision framing for the networks you just hand-wrote. |
| [03 · tokenization](modules/03-tokenization) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 6 | The full tokenizers library — training a tokenizer, fast tokenizers, and the pipeline behind the BPE you built by hand. |
| [04 · attention-transformer](modules/04-attention-transformer) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 1 ("how Transformers work") | The conceptual companion to your char-GPT: attention, architectures, and the encoder/decoder families. |
| [05 · transformers-library](modules/05-transformers-library) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 1–4 | `pipeline`, `AutoModel`, fine-tuning basics — the canonical tour of the library whose anatomy you dissected. |
| [05½ · data-evaluation](modules/05a-data-evaluation) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 5 · [Evaluate](https://huggingface.co/docs/evaluate) | Turn a working model into an honest experiment: leakage-safe splits, thresholds, calibration, slices, and an evaluation card. |
| [06 · fine-tuning](modules/06-fine-tuning) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 11–12 · **[smol-course](https://huggingface.co/learn/smol-course)** | The main sequel. LLM ch. 11 is SFT / chat templates / LoRA / eval with TRL; ch. 12 is RL and GRPO (Open R1). smol-course is the deep, project-based post-training track. |
| [07 · inference-internals](modules/07-inference-internals) | llama.cpp / GGUF ecosystem · [CV Course](https://huggingface.co/learn/computer-vision-course) unit 9 (model optimization) | No single HF course owns inference internals; the GGUF/llama.cpp world and the CV optimization unit (quantization, distillation, pruning) are the closest continuations of your KV-cache and quantization work. |
| [08 · vision](modules/08-vision) | [Community Computer Vision Course](https://huggingface.co/learn/computer-vision-course) | The full breadth: CNNs, ViTs, multimodal, generative vision, and the task zoo beyond the single ViT you fine-tuned. |
| [09 · diffusion](modules/09-diffusion) | [Diffusion Course](https://huggingface.co/learn/diffusion-course) | From your toy denoiser to DDPMs, guidance, Stable Diffusion, and the `diffusers` library end to end. |
| [10 · agents](modules/10-agents) | [Agents Course](https://huggingface.co/learn/agents-course) · [MCP Course](https://huggingface.co/learn/mcp-course) · [Context Course](https://huggingface.co/learn/context-course) | The whole agentic stack: agent frameworks and certification, the Model Context Protocol in depth, and context engineering for code agents. |

---

## The full catalog

Every course currently on [huggingface.co/learn](https://huggingface.co/learn),
with an honest status note where it matters.

### LLM Course

The flagship. A 12-chapter path from Transformer fundamentals to fine-tuning and
modern reasoning models, built on the `transformers`, `datasets`, and `tokenizers`
libraries.

- **Chapters 1–4** — Transformer fundamentals: how models work, using `transformers`, fine-tuning.
- **Chapters 5–8** — the `datasets` library, the `tokenizers` library, and classic NLP tasks.
- **Chapter 9** — building and sharing demos with Gradio.
- **Chapters 10–12** — advanced: curating datasets, supervised fine-tuning with TRL (chat templates, LoRA, evaluation), and "Open R1 for Students" (RL intro, the DeepSeek R1 paper, GRPO in TRL).

Prerequisites: comfortable Python; basic deep-learning familiarity helps. Hands-on
via Colab notebooks throughout. No formal certificate, but it is the backbone of
the platform.

### smol-course (v2)

The deep post-training track — hands-on fine-tuning of small models (SmolLM3,
SmolVLM2) with TRL and PEFT, designed to run on a local machine. This is the main
sequel to our module 06.

- Instruction Tuning — SFT, chat templates, instruction following.
- Evaluation — benchmarks and custom domain evaluation.
- Preference Alignment — aligning to human preferences with DPO.
- Vision Language Models — adapting and using multimodal models.
- Reinforcement Learning — policy-based optimization.
- Synthetic Data — generating datasets for custom domains.
- Award Ceremony — final projects and showcase.

Prerequisites: the LLM Course (or our modules 05–06) and working PyTorch. Very
hands-on, local-first. **Two free certificates:** a Fundamentals certificate (unit
1) and a Certificate of Completion (all units + final project).

### Agents Course

Build and ship AI agents, from first principles to a graded final assignment.

- Units 0–4 — fundamentals, agent frameworks (smolagents, LangGraph, LlamaIndex), real use cases, and a final hands-on assignment.
- Bonus units — fine-tuning a model for function-calling, agent observability and evaluation, and agents in games.

Prerequisites: Python and basic LLM familiarity. Hands-on with runnable notebooks
and a leaderboard-graded final project. **Two free certifications** (fundamentals
and a completion certificate).

### MCP Course

The Model Context Protocol in depth — the open standard for connecting tools and
data to models and agents.

- MCP fundamentals and architecture.
- End-to-end and deployed use cases.
- Examples in both Python and TypeScript.

Prerequisites: general programming; the Agents Course helps but is not required.
Hands-on. **Free certifications** available.

### Context Course

Context engineering for code agents, built on the Claude Agent SDK / Claude Code.
The newest addition to the platform.

- Unit 0 — onboarding and setup.
- Unit 1 — building and sharing agent skills.
- Unit 2 — the Model Context Protocol for connecting tools.
- Unit 3 — plugins and workflow design.
- Unit 4 — multi-agent patterns and coordination.
- Units 5–6 — lifecycle hooks and building a minimal agent loop.

Prerequisites: comfort with agents and a coding agent. Hands-on. **Two free
certificates:** a Fundamentals certificate (units 1–2) and an Engineering
certificate (all units + capstone).

### Community Computer Vision Course

A broad, community-written tour of modern computer vision in 13 units.

- Fundamentals; CNNs; Vision Transformers; multimodal models (CLIP); generative
  vision; the task zoo; video; 3D vision; model optimization; synthetic data;
  zero-shot; ethics; and current trends.

Prerequisites: deep-learning basics and PyTorch. Hands-on notebooks per unit. No
formal certificate. Solid and comprehensive; breadth over polish, as befits a
community effort.

### Diffusion Course

Denoising diffusion models with the `diffusers` library, in four units.

- Unit 1 — introduction to diffusion models.
- Unit 2 — fine-tuning and guidance.
- Unit 3 — Stable Diffusion.
- Unit 4 — going further with diffusion.

Prerequisites: PyTorch and comfort with training loops. Hands-on notebooks.
**Certificate** available (see the course FAQ). Older material but still solid and
accurate.

### Deep RL Course

Reinforcement learning from theory to hands-on agents, pairing readings with Colab
practice on Stable-Baselines3, CleanRL, and Sample Factory.

- Q-learning and value methods; policy gradients; Actor-Critic; PPO; multi-agent
  RL; and advanced topics across the units.

Prerequisites: Python and solid deep-learning basics. Hands-on Colabs.
**Certificate** available. **Status note:** this course is in low-maintenance mode.
Unit 7's AI-vs-AI leaderboard is no longer operational, so that head-to-head
exercise cannot be completed as written — the rest of the course and the
certification still work.

### Audio Course

Audio machine learning end to end, in seven units.

- Unit 1 — working with audio data.
- Unit 2 — audio pipelines and applications.
- Unit 3 — Transformer architectures for audio.
- Unit 4 — building a music-genre classifier.
- Unit 5 — automatic speech recognition and meeting transcription.
- Unit 6 — text-to-speech.
- Unit 7 — putting it together into real applications.

Prerequisites: the LLM Course (or equivalent Transformer familiarity) helps.
Hands-on. Older material but well-structured and still solid.

### Robotics Course

**New and in progress.** Real-world robotics with LeRobot.

- The classical-robotics units are done; units on reinforcement learning,
  imitation learning, and robot foundation models are still being written.

Prerequisites: Python and RL basics. Hands-on with LeRobot. Check back as units
land — the syllabus is not yet complete.

### Open-Source AI Cookbook

A growing collection of practical, task-focused recipe notebooks rather than a
chaptered course — RAG, agents, fine-tuning, structured generation, and more. Dip
in for a specific technique. No certificate; not meant to be read in order.

### ML for Games and ML for 3D

Two lighter, older tracks: [Machine Learning for
Games](https://huggingface.co/learn/ml-games-course) and [Machine Learning for
3D](https://huggingface.co/learn/ml-for-3d-course). Narrower in scope and less
actively maintained than the courses above, but useful if those domains are your
target.

---

## Certifications you can collect

Several HF courses grant a free certificate that shows on your Hugging Face
profile:

- **[Agents Course](https://huggingface.co/learn/agents-course)** — two: a fundamentals certificate and a completion certificate.
- **[MCP Course](https://huggingface.co/learn/mcp-course)** — free certification.
- **[Context Course](https://huggingface.co/learn/context-course)** — two: a Fundamentals certificate and an Engineering certificate (with capstone).
- **[smol-course](https://huggingface.co/learn/smol-course)** — two: a Fundamentals certificate and a Certificate of Completion.
- **[Deep RL Course](https://huggingface.co/learn/deep-rl-course)** — certificate available (course in low-maintenance mode; see the status note above).
- **[Diffusion Course](https://huggingface.co/learn/diffusion-course)** — certificate available (see the course FAQ).

---

## Suggested continuation paths

Three routes out of this course, each stitched from HF Learn courses.

**The LLM path.** You liked module 06 and want to master post-training. Do the
[LLM Course](https://huggingface.co/learn/llm-course) chapters 11–12, then
**[smol-course](https://huggingface.co/learn/smol-course)** end to end: instruction
tuning → DPO preference alignment → GRPO and Open R1 for RL-based reasoning. Finish
with the smol-course completion certificate.

**The multimodal path.** You want breadth across senses. Start with the
[Community Computer Vision Course](https://huggingface.co/learn/computer-vision-course),
add the [Audio Course](https://huggingface.co/learn/audio-course) for speech and
sound, then close the loop with the Vision Language Models module of
**[smol-course](https://huggingface.co/learn/smol-course)** to fine-tune a
multimodal model yourself.

**The agentic path.** You want models that act. Do the
[Agents Course](https://huggingface.co/learn/agents-course) (and earn both
certificates), go deep on tool-and-data plumbing with the
[MCP Course](https://huggingface.co/learn/mcp-course), then learn to engineer the
context that makes code agents effective in the
[Context Course](https://huggingface.co/learn/context-course).

---

# Part 2 · Beyond Hugging Face

The best free material on the open web, grouped the way you would actually reach
for it. Every URL below was fetched and checked in **July 2026**; the `Cost`
column is honest about freemium and paid-book cases, and the `Modules` column
says which of our modules a resource enriches. Several famous resources are
frozen, archived, or have moved — those caveats are stated inline and collected
in [Link rot notes](#link-rot-notes) at the end.

Nothing here duplicates Part 1: no item below is an HF Learn course.

## The five must-reads

If you read nothing else in Part 2, read these.

1. **[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)** (Polo Club)
   A real GPT-2 small running client-side, with live numbers at every stage from
   embeddings through attention to softmax. It is the proof-of-concept for this
   course's whole explorables approach — study its abstraction-level transitions
   before you touch module 04's diagrams.

2. **[Google PAIR AI Explorables](https://pair.withgoogle.com/explorables/)** + **[VISxAI Hall of Fame](https://visxai.io/index.html)**
   Taken as a pair, the two highest-yield sources of explorable *design patterns*
   on the open web — and unlike Distill, both are still producing new work in 2026.

3. **[Stanford CS336 — Language Modeling from Scratch](http://cs336.stanford.edu/)**
   The one public university course whose assignment ladder mirrors ours almost
   exactly: tokenizer, model, Triton/FlashAttention, scaling laws, data pipeline,
   RL. Videos and repos free, self-study explicitly invited. The natural "go
   deeper" target for modules 03, 04 and 07.

4. **[labml.ai — Annotated Paper Implementations](https://nn.labml.ai/)**
   One actively-maintained site with ~60+ side-by-side annotated PyTorch
   implementations spanning modules 04, 06, 07, 08 and 09. The highest
   coverage-per-link on this entire page.

5. **[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)** (Google DeepMind)
   Free, figure-heavy, and the clearest treatment anywhere of the arithmetic that
   makes module 07's Zig engine fast or slow — rooflines, memory bandwidth,
   KV-cache economics — worked through on real LLaMA-3 numbers.

---

## Further study, module by module

The short list. Two to four external picks per module, chosen rather than dumped
— the full sets are in the category tables below.

| Our module | Best external picks |
|---|---|
| [00a · perceptron](modules/00a-perceptron) | [Why Machines Learn](https://anilananthaswamy.com/why-machines-learn) (the perceptron and least-squares chapters) · [MLU-Explain](https://mlu-explain.github.io/) linear regression · [The Matrix Calculus You Need](https://explained.ai/matrix-calculus/index.html) |
| [00b · bayes-knn-pca](modules/00b-bayes-knn-pca) | [Why Machines Learn](https://anilananthaswamy.com/why-machines-learn) (Bayes, nearest neighbours, eigenvectors) · [Mathematics for Machine Learning](https://mml-book.github.io/) ch. 6 and 10 · [Explained Visually](https://setosa.io/ev/) PCA · [Immersive Linear Algebra](https://immersivemath.com/ila/index.html) |
| [00c · kernels-hopfield](modules/00c-kernels-hopfield) | [Why Machines Learn](https://anilananthaswamy.com/why-machines-learn) (kernels, Hopfield, the modern bridge) · [MLU-Explain](https://mlu-explain.github.io/) bias-variance and double descent · ["Hopfield Networks is All You Need"](https://arxiv.org/abs/2008.02217) · [Mathematics for Machine Learning](https://mml-book.github.io/) ch. 12 |
| [01 · autograd](modules/01-autograd) | [micrograd](https://github.com/karpathy/micrograd) (the reference target) · [Karpathy Zero to Hero](https://karpathy.ai/zero-to-hero.html) lectures 1–2 · [Backprop Explainer](https://xnought.github.io/backprop-explainer/) · [The Matrix Calculus You Need](https://explained.ai/matrix-calculus/index.html) |
| [02 · neural-networks](modules/02-neural-networks) | [TensorFlow Playground](https://playground.tensorflow.org/) · [MLU-Explain](https://mlu-explain.github.io/) (neural networks, bias-variance, double descent) · [3Blue1Brown](https://www.3blue1brown.com/) NN series · [Dive into Deep Learning](https://d2l.ai/) |
| [03 · tokenization](modules/03-tokenization) | [Cornell BPE-vs-WordPiece visualizer](https://www.cs.cornell.edu/courses/cs4782/2026sp/demos/bytepair) · [Tiktokenizer](https://tiktokenizer.vercel.app/) · Karpathy's "Let's build the GPT Tokenizer" ([Zero to Hero](https://karpathy.ai/zero-to-hero.html)) · [CS336](http://cs336.stanford.edu/) assignment 1 |
| [04 · attention-transformer](modules/04-attention-transformer) | [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) · [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) · [LLM Visualization](https://bbycroft.net/llm) (3D inference path) · [The Illustrated Transformer](https://jalammar.github.io/) |
| [05 · transformers-library](modules/05-transformers-library) | [Hands-On LLM notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) · [Build a LLM (From Scratch) code](https://github.com/rasbt/LLMs-from-scratch) · [mlabonne LLM Course](https://github.com/mlabonne/llm-course) roadmaps · [Embedding Atlas](https://apple.github.io/embedding-atlas/) |
| [06 · fine-tuning](modules/06-fine-tuning) | [nanochat](https://github.com/karpathy/nanochat) (tokenizer → pretrain → SFT → eval, one repo) · [CS336](http://cs336.stanford.edu/) assignment 5 (alignment + reasoning RL) · [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) bonus LoRA material · [CleanRL](https://docs.cleanrl.dev/) |
| [07 · inference-internals](modules/07-inference-internals) | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) · [Inside vLLM](https://www.aleksagordic.com/blog/vllm) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [llm.c](https://github.com/karpathy/llm.c) (has a Zig port) · [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) |
| [08 · vision](modules/08-vision) | [CNN Explainer](https://poloclub.github.io/cnn-explainer/) · [CS231n lecture notes](https://cs231n.stanford.edu/) · [labml.ai](https://nn.labml.ai/) ViT implementation · [Explained Visually](https://setosa.io/ev/) image kernels |
| [09 · diffusion](modules/09-diffusion) | [Diffusion Explainer](https://poloclub.github.io/diffusion-explainer/) · [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) · [Step-by-Step Diffusion tutorial](https://arxiv.org/abs/2406.08929) · [fast.ai Part 2](https://course.fast.ai/) (diffusion from foundations) |
| [10 · agents](modules/10-agents) | [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) · [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) · [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) · [Lil'Log](https://lilianweng.github.io/) |

---

## Interactive and visual explainers

The category that matters most for this course's own in-browser explorables —
both as design inspiration and as further viewing.

| Resource | URL | What it is | Cost | Modules |
|---|---|---|---|---|
| **Transformer Explainer** (Polo Club) | https://poloclub.github.io/transformer-explainer/ | Live GPT-2 small (124M) running in-browser via ONNX Runtime; step through embeddings → attention → softmax with live numeric updates and adjustable temperature. CHI 2026 paper, 490k+ users. | Free | 04, 05 |
| **LLM Visualization** (Brendan Bycroft) | https://bbycroft.net/llm | 3D walkthrough of a single token's inference path through nano-GPT / GPT-2 / GPT-3, every matmul rendered as geometry. | Free | 04, 07 |
| **CNN Explainer** (Polo Club) | https://poloclub.github.io/cnn-explainer/ | Interactive CNN layer by layer — convolution windows, ReLU, pooling — on a live TinyVGG. | Free | 02, 08 |
| **Diffusion Explainer** (Polo Club) | https://poloclub.github.io/diffusion-explainer/ | Stable Diffusion explained visually: prompt → text encoder → iterative denoising, with interactive guidance-scale comparison. | Free | 09 |
| **GAN Lab** (Polo Club) | https://poloclub.github.io/ganlab/ | Train a GAN in-browser (TensorFlow.js): generator manifold as a warped grid, discriminator as a heatmap, slow-motion and step-by-step modes. | Free | 02, 09 |
| **TensorFlow Playground** | https://playground.tensorflow.org/ | The canonical in-browser NN sandbox — pick a dataset, add layers, watch weights and the decision boundary train live. Apache-licensed, built to be forked. | Free | 02 |
| **Google PAIR — AI Explorables** | https://pair.withgoogle.com/explorables/ | ~16 interactive essays from Google Research: sparse autoencoders, patchscopes, grokking, model confidence, out-of-distribution data, fairness, differential privacy. Actively maintained. | Free | general, 04, 06 |
| **VISxAI "Hall of Fame"** | https://visxai.io/index.html | The workshop that exists to solicit explainables about AI; the Hall of Fame indexes award-winning submissions 2018–2025 (9th edition Oct 2026). Best single place to mine explorable ideas. | Free | general |
| **Distill.pub** | https://distill.pub/ | The gold standard for interactive ML articles. **On hiatus since July 2021** — archive fully live and still the best writing on feature visualization, interpretability, momentum, GNNs. | Free | general, 02, 08 |
| **Distill — A Gentle Introduction to GNNs** | https://distill.pub/2021/gnn-intro/ | Named exemplar: hoverable message-passing diagram plus a live TF.js "GNN Playground" for molecule prediction. A reference implementation for explorable UX. | Free | general |
| **MLU-Explain** (Amazon MLU) | https://mlu-explain.github.io/ | 16 scroll-driven visual explainers — neural networks, linear/logistic regression, trees, random forests, ROC/AUC, cross-validation, bias-variance, double descent. Beautiful D3, source on GitHub. | Free | 02, general |
| **colah's blog** (Chris Olah) | https://colah.github.io/ | "Understanding LSTM Networks", "Neural Networks, Manifolds and Topology", "Calculus on Computational Graphs: Backpropagation", "Conv Nets: A Modular Perspective", the Circuits series. The intellectual ancestor of every good ML explainer. | Free | 01, 02, 04, 08 |
| **Jay Alammar — Illustrated series** | https://jalammar.github.io/ | The Illustrated Transformer / GPT-2 / BERT / Word2vec / Stable Diffusion / Retrieval Transformer. **Blog is frozen** — new work moved to Substack (next row) — but all the classics remain live. | Free | 03, 04, 05, 09 |
| **Jay Alammar — Substack** | https://newsletter.languagemodels.co/ | Where the illustrated series continues: The Illustrated DeepSeek-R1 (Jan 2025), The Illustrated GPT-OSS (Aug 2025), NeurIPS 2025 embedding map. Key illustrated posts free. | Free (freemium) | 04, 05, 06 |
| **Transformer Circuits Thread** (Anthropic) | https://transformer-circuits.pub/ | Interactive interpretability research articles — verbalizable representations as a global workspace (Jul 2026), natural-language autoencoders, emotion concepts in Sonnet 4.5, attention-computation tracing. Dense, but the interactive figures are exemplary. | Free | 04, 07 |
| **Neuronpedia** | https://neuronpedia.org/ | Open-source interpretability platform — browse and steer model internals, Circuit Tracer, 5+ TB of free activations and feature explanations, free API and Python/TS libs. | Free | 04, 07 |
| **Backprop Explainer** | https://xnought.github.io/backprop-explainer/ | Small interactive explainer for backpropagation and gradient flow. The closest existing analogue to a module-01 explorable. | Free | 01 |
| **Tokenization visualizer** (Cornell CS4782) | https://www.cs.cornell.edu/courses/cs4782/2026sp/demos/bytepair | Step-through BPE **vs WordPiece** side by side: watch the merge table build, then test the learned vocab on new text. Exactly the shape of a module-03 explorable. | Free | 03 |
| **Tiktokenizer** | https://tiktokenizer.vercel.app/ | Paste text, see live token boundaries and counts across tokenizers (GPT-4o and others), whitespace toggle. Open source. The fastest "tokenizers are weird" demo. | Free | 03 |
| **Seeing Theory** (Brown) | https://seeing-theory.brown.edu/ | Six chapters of D3 probability and statistics visualizations — distributions, CLT, frequentist vs Bayesian inference, regression. **Banner says archived for reference**; the interactives all still work. | Free | general, 09 |
| **Explained Visually** (setosa.io) | https://setosa.io/ev/ | Bret-Victor-inspired interactives: image kernels, PCA, eigenvectors, OLS, Markov chains, conditional probability. **Last updated Feb 2015** — dated, but the image-kernels and eigenvector pieces are still the clearest around. | Free | 08, general |
| **Dodrio** (Polo Club) | https://poloclub.github.io/dodrio/ | Compares transformer attention heads against linguistic and syntactic knowledge. Niche; useful as a "what else can you show about attention" reference. | Free | 04 |
| **Explorable Explanations directory** | https://explorable-explanations.com/en/ | Curated directory of interactive articles with an explicit "AI Models" section. Good for idea-mining; quality of listed items is uneven — treat it as an index, not a recommendation. | Free | general |

---

## From-scratch courses and series

| Resource | URL | What it is | Cost | Modules |
|---|---|---|---|---|
| **Karpathy — Neural Networks: Zero to Hero** | https://karpathy.ai/zero-to-hero.html | 8 lectures (~14h): micrograd → makemore → MLP → BatchNorm → backprop ninja → WaveNet → "Let's build GPT" → "Let's build the GPT Tokenizer". Still marked ongoing. The closest thing to this course's spine. | Free | 01, 02, 03, 04 |
| **micrograd** | https://github.com/karpathy/micrograd | ~100-line scalar autograd plus a 50-line NN lib with a PyTorch-like API. The reference target for module 01. | Free | 01 |
| **nanochat** | https://github.com/karpathy/nanochat | Full pipeline in one minimal repo: tokenizer → pretrain → SFT → eval → CLI chat. A single `--depth` knob sets all hyperparameters. ~$48 on 8×H100 (~2h) for GPT-2-class, ~$15 spot. | Free (code) | 03, 04, 05, 06 |
| **nanoGPT** | https://github.com/karpathy/nanoGPT | ~300-line GPT training script that reproduced GPT-2 124M. **Explicitly deprecated as of Nov 2025** — the README points to nanochat. Still the cleanest single-file read; treat it as historical reference, not a starting point. | Free | 04 |
| **llm.c** | https://github.com/karpathy/llm.c | GPT-2/GPT-3 training in raw C/CUDA, no PyTorch. ~1000-line CPU version plus optimized kernels, currently ~7% faster than PyTorch nightly. Has **Zig**, Rust and Go ports — directly relevant to our Zig engine. | Free | 07 |
| **Stanford CS336 — Language Modeling from Scratch** | http://cs336.stanford.edu/ | Spring 2026 is current. Five public assignments: basics (tokenizer/model/optimizer), systems (Triton + FlashAttention2, distributed), scaling laws, Common Crawl data pipeline, alignment + reasoning RL. Full YouTube playlist. Explicitly welcomes self-study. | Free | 03, 04, 06, 07 |
| **ARENA curriculum** | https://learn.arena.education/ | Chapters 0–4: fundamentals/CNNs/backprop/generative, transformer interpretability (13 sections, TransformerLens), RL including RLHF, LLM evals with Inspect, alignment science. Free and open-access; program applications are separate. | Free | 01, 02, 04, 06, 10 |
| **ARENA_3.0 repo** | https://github.com/callummcdougall/ARENA_3.0 | The exercises and notebooks behind the curriculum above, with Colab links and an install script. | Free | 01, 02, 04, 10 |
| **fast.ai — Practical Deep Learning for Coders** | https://course.fast.ai/ | Part 1 (9 lessons: vision, NLP, tabular, collaborative filtering) plus Part 2 "Deep Learning Foundations to Stable Diffusion" (25+ lessons). Part 2 is the standout — it builds diffusion up from scratch. Free book online; runs on Kaggle free tiers. | Free | 02, 08, 09 |
| **Dive into Deep Learning (d2l.ai)** | https://d2l.ai/ | v1.0.3. Interactive textbook where every chapter is a runnable notebook, in PyTorch / JAX / TF / NumPy simultaneously. Adopted at 500 universities. Free online; Cambridge print edition paid. | Free | 02, 04, 08, general |
| **Understanding Deep Learning** (Simon Prince) | https://udlbook.github.io/udlbook/ | Free-PDF textbook with per-chapter Python notebooks plus downloadable figures and slides. The figure set is unusually good for teaching material. | Free (PDF) | 02, 04, 08, 09 |
| **Build a LLM (From Scratch)** — code | https://github.com/rasbt/LLMs-from-scratch | Sebastian Raschka's repo, 100k+ stars: 7 chapters from text data → attention → GPT → pretraining → classification finetune → instruction finetune, plus bonus LoRA / Llama / Qwen / Gemma / RL material. Runs on a laptop. Code free; book paid (Manning). | Free (code) | 03, 04, 05, 06 |
| **MIT 6.S191 — Intro to Deep Learning** | http://introtodeeplearning.com/ | 2026 edition ran Mar–May 2026. All lectures on YouTube, all labs open-sourced under MIT license. The fastest credible survey — vision, robotics, language, generative. | Free | 02, 08, 09, general |
| **Stanford CS231n** | https://cs231n.stanford.edu/ | Spring 2026 offering live. The legendary lecture notes remain public at `cs231n.github.io`; prior-year videos on YouTube. Current-quarter videos and assignments are Canvas-gated. | Free (notes) | 02, 08 |
| **Stanford CS224n** | https://web.stanford.edu/class/cs224n/ | Winter 2026 offering live. Slides public per lecture, notes public for roughly the first half, 2024 lecture videos free on YouTube. Current videos Canvas-gated. (XCS224N is the paid version.) | Free (slides) | 03, 04, 05 |
| **Cornell CS4782 — Intro to Deep Learning** | https://www.cs.cornell.edu/courses/cs4782/2026sp/ | Spring 2026. Student-compiled lecture notes public and — the reason to bookmark it — a set of public in-browser `demos/`, including the BPE/WordPiece one above. Assignments enrollment-gated. | Free (partial) | 02, 03, 04 |
| **Umar Jamil — from-scratch video series** | https://umarjamil.org/videos | Long-form "code it from scratch in PyTorch" videos: Transformer, LLaMA 2, Stable Diffusion, Flash Attention with Triton, multimodal VLM. Slides and code on GitHub (`hkproj`). Site last updated Sep 2024; videos free on YouTube. | Free | 04, 06, 07, 08, 09 |
| **mlabonne — LLM Course** | https://github.com/mlabonne/llm-course | 81k stars. Three tracks (fundamentals / LLM Scientist / LLM Engineer) with visual roadmaps and Colab notebooks for finetuning, quantization, model merging. The best "what next, and in what order" map. | Free | 05, 06, 07 |
| **Hands-On Large Language Models** — notebooks | https://github.com/HandsOnLLM/Hands-On-Large-Language-Models | Jay Alammar + Maarten Grootendorst, O'Reilly 2024, ~300 custom illustrations. All 12 chapters' notebooks free and Colab-ready (Apache-2.0); the book itself is paid. | Free (code) | 03, 05, 06 |

---

## RL and agents

| Resource | URL | What it is | Cost | Modules |
|---|---|---|---|---|
| **OpenAI Spinning Up in Deep RL** | https://spinningup.openai.com/en/latest/ | Still the best on-ramp: RL intro essays, "spinning up as a deep RL researcher", exercises, and clean implementations of VPG, TRPO, PPO, DDPG, TD3, SAC in PyTorch and TF. Long-term-support mode, not actively expanded. | Free | 06, 10 |
| **CleanRL** | https://docs.cleanrl.dev/ | Single-file, dependency-light implementations of 13+ algorithms (PPO, DQN, C51, DDPG, SAC, TD3, PPG, RND, RPO, PPO-TrXL, PQN, Rainbow, QDagger), benchmarked across 34+ games with tracked runs. The "read the whole algorithm in one file" library. | Free | 06, 10 |
| **Sutton & Barto — Reinforcement Learning: An Introduction (2e)** | http://incompleteideas.net/book/the-book-2nd.html | The canonical RL textbook, free PDF from Sutton's own site, plus errata, code and teaching aids. **The HTTPS certificate is self-signed** — fine in a browser, breaks automated fetchers. | Free (PDF) | 06, 10 |
| **Anthropic — Building Effective Agents** | https://www.anthropic.com/engineering/building-effective-agents | Dec 2024. The pattern vocabulary everyone now uses: augmented LLM, prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer, then autonomous agents. Argues hard for simplicity over frameworks. | Free | 10 |
| **Anthropic — Effective Context Engineering for AI Agents** | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | Sep 2025. Context as a finite resource: context rot, tool design, just-in-time retrieval, compaction, structured note-taking, sub-agents. The essential companion to the above. | Free | 10 |
| **12-Factor Agents** | https://github.com/humanlayer/12-factor-agents | 24.9k stars, 2025. Twelve production principles — own your prompts, own your context window, tools as structured output, stateless reducers, explicit human-contact points, compact errors. Framework-skeptical and concrete. | Free | 10 |
| **Lil'Log** (Lilian Weng) | https://lilianweng.github.io/ | Still active: "Harness Engineering for Self-Improvement" (Jul 2026), "Scaling Laws, Carefully" (Jun 2026), "Why We Think" (2025), plus the reference posts on LLM-powered autonomous agents, reward hacking, and diffusion. | Free | 09, 10, general |

---

## Math foundations

| Resource | URL | What it is | Cost | Modules |
|---|---|---|---|---|
| **3Blue1Brown** | https://www.3blue1brown.com/ | Essence of Linear Algebra, Essence of Calculus, and the Neural Networks series — which now extends through GPT, attention and transformers and, as of Jul 2026, a cross-entropy and "compression is intelligence" arc. Manim-rendered; the visual bar our figures should aim at. | Free (Patreon optional) | 01, 02, 04, general |
| **Why Machines Learn: The Elegant Math Behind Modern AI** | https://anilananthaswamy.com/why-machines-learn | Anil Ananthaswamy, Dutton 2024. A narrative history of the mathematics — perceptrons and their convergence proof, least squares, Bayes, nearest neighbours, eigenvectors and PCA, kernels and SVMs, Hopfield networks, and the bias-variance story up to double descent — told through the people who found it, with the derivations kept honest. The book Track 0 is patterned on: read it beside those modules, which rebuild the same ideas in original code and figures. | Paid (book) | 00a, 00b, 00c primarily · 02, 04, 08 |
| **Mathematics for Machine Learning** | https://mml-book.github.io/ | Deisenroth, Faisal & Ong, CUP 2020. Part I: linear algebra, analytic geometry, matrix decompositions, vector calculus, probability, optimization. Part II: regression, PCA, GMMs, SVMs. Jupyter tutorials and solutions. The authors commit to keeping the PDFs free. | Free (PDF) | general, 01, 02 |
| **The Matrix Calculus You Need For Deep Learning** | https://explained.ai/matrix-calculus/index.html | Parr & Howard. Exactly the Jacobian and chain-rule vocabulary needed to read a backward pass, and nothing more. Printable copy at https://arxiv.org/abs/1802.01528. | Free | 01 |
| **Immersive Linear Algebra** | https://immersivemath.com/ila/index.html | Ström, Åström & Akenine-Möller — "the world's first linear algebra book with fully interactive figures", 10 chapters through eigenvalues. Another strong interactive-figure design reference. | Free to read online | general, 01 |

---

## Papers as pedagogy

Annotated implementations — paper text interleaved with runnable code. The format
this course's algorithm cards are related to, and worth studying as a genre.

| Resource | URL | What it is | Cost | Modules |
|---|---|---|---|---|
| **The Annotated Transformer** | https://nlp.seas.harvard.edu/annotated-transformer/ | Harvard NLP's line-by-line PyTorch reimplementation of "Attention Is All You Need". The archetype of the genre. | Free | 04 |
| **labml.ai — Annotated Paper Implementations** | https://nn.labml.ai/ | Actively maintained, ~60+ side-by-side annotated PyTorch implementations: transformers (ViT, GPT, Flash Attention, RoPE), DDPM/DDIM, Stable Diffusion, StyleGAN2, GANs, PPO/DQN, GATs, U-Net, LSTM, optimizers, normalization layers. The best breadth-per-click resource on this page. | Free | 04, 06, 07, 08, 09 |
| **The Annotated Diffusion Model** | https://huggingface.co/blog/annotated-diffusion | Rogge & Rasul, Jun 2022. DDPM from scratch in PyTorch — variance schedules, reparameterization, U-Net with ResNet blocks/attention/position embeddings, training on Fashion-MNIST, sampling. An HF-hosted blog post, not an HF Learn course; still the canonical annotated diffusion walkthrough. | Free | 09 |
| **Step-by-Step Diffusion: An Elementary Tutorial** | https://arxiv.org/abs/2406.08929 | Nakkiran, Bradley, Zhou & Advani. 35 pages, 11 figures, CC BY-NC-SA. An accessible first course on diffusion *and flow matching* assuming no diffusion background — the theory companion to the annotated implementation above. | Free | 09 |
| **The Annotated S4** | https://srush.github.io/annotated-s4/ | Sasha Rush's annotated JAX implementation of Structured State Spaces. Worth linking as the "what came after attention" branch, and as a second worked example of the format. | Free | 04, 07 |
| **GPT in 60 Lines of NumPy** | https://jaykmody.com/blog/gpt-from-scratch/ | Jay Mody. A full GPT-2 forward pass in 60 lines of NumPy (120 commented), loading OpenAI's released weights and verified against the official repo. The clearest proof that inference is just matmuls. | Free | 04, 07 |
| **Transformers from Scratch** (Brandon Rohrer) | https://brandonrohrer.com/transformers.html | Builds attention from one-hot vectors and dot products upward, with toy vocabularies, assuming no matrix calculus. | Free | 04 |

---

## Graphical-first and in-browser

The 2025–2026 material on running and visualizing real models client-side, plus
the systems reading behind module 07.

| Resource | URL | What it is | Cost | Modules |
|---|---|---|---|---|
| **Transformers.js — WebGPU guide** | https://huggingface.co/docs/transformers.js/en/guides/webgpu | The one-line switch (`device: 'webgpu'`) that puts real models on the GPU in-browser via ONNX Runtime Web. Direct blueprint for in-browser experiments. | Free | 05, 07, 08 |
| **webml-community Spaces** | https://huggingface.co/spaces/webml-community | The live gallery of what browsers can now run client-side: a 1-bit 27B LLM on WebGPU, Gemma/Llama/Qwen variants, in-browser image generation, SOTA audio transcription, TTS, PII masking, video captioning, a 3D "Semantic Galaxy" embedding explorer. The best proof-of-possible for module 07. | Free | 05, 07, 08, 09 |
| **WebLLM** (MLC) | https://webllm.mlc.ai/ | In-browser LLM inference engine on WebGPU with an OpenAI-compatible API; supports Llama, Phi, Gemma. Live demo at chat.webllm.ai. The alternative runtime to compare against Transformers.js. | Free | 07 |
| **Embedding Atlas** (Apple) | https://apple.github.io/embedding-atlas/ | MIT-licensed WebGPU tool that renders and cross-filters millions of embeddings with automatic clustering and live search. Runs in the browser — excellent for a "look at your own embedding space" exercise. | Free | 03, 05, 08 |
| **How to Scale Your Model** (Google DeepMind) | https://jax-ml.github.io/scaling-book/ | Feb 2025, 12 chapters, ten DeepMind authors. Rooflines, TPU/GPU architecture, matmul arithmetic, the transformer math, training and inference parallelism worked through on LLaMA-3, then JAX profiling. The systems-thinking backbone for module 07. | Free | 06, 07 |
| **Inside vLLM: Anatomy of a High-Throughput Inference System** | https://www.aleksagordic.com/blog/vllm | Aug 2025, pinned to a specific commit. Scheduling, paged attention, continuous batching, chunked prefill, prefix caching, FSM-guided decoding, speculative decoding, multi-GPU, then benchmarking latency vs throughput. Exactly the map module 07's Zig engine re-derives. | Free | 07 |
| **llama.cpp** | https://github.com/ggml-org/llama.cpp | 122k stars. Dependency-free C/C++ inference, 1.5–8-bit quantization, Metal/CUDA/HIP/SYCL backends, `llama-cli` and an OpenAI-compatible `llama-server`. The reference implementation to read alongside a hand-rolled engine. | Free | 07 |
| **Google — Machine Learning Crash Course** | https://developers.google.com/machine-learning/crash-course | The refreshed edition, with animated videos, interactive visualizations and exercises, plus newer modules on AutoML and intro-to-LLMs. Uneven depth for this audience, but the interactive widgets are worth studying as UI patterns. | Free | general, 02 |

---

## Link rot notes

Verified July 2026. If a link above surprises you, the reason is probably here.

- **Distill.pub** — on hiatus since 2 Jul 2021; the archive is fully live. Cite it as an archive.
- **Jay Alammar** — `jalammar.github.io` is frozen; new work is at `newsletter.languagemodels.co`. Link both.
- **nanoGPT → nanochat** — nanoGPT's README declares it deprecated (Nov 2025) in favour of **nanochat**. Any "start here" pointer should go to nanochat.
- **`e2eml.school/transformers.html`** → 301 → `brandonrohrer.com/transformers.html`.
- **`stanford-cs336.github.io/spring2025/`** → 301 → `cs336.stanford.edu`; the Spring 2025 site is explicitly labelled archived. Use the Spring 2026 root.
- **`ggerganov/llama.cpp`** → moved to the `ggml-org` org: `ggml-org/llama.cpp`.
- **ARENA** — the old `arena-chapter*.streamlit.app` URLs now redirect to a Streamlit auth wall. The free curriculum lives at `learn.arena.education`.
- **Seeing Theory** carries an "archived for reference" banner (the interactives still work); **setosa.io/ev** has not been updated since Feb 2015.
- **incompleteideas.net** (Sutton & Barto) serves a self-signed TLS certificate — fine in a browser, breaks automated fetches.
- **FT — "Generative AI exists because of the transformer"** (`ig.ft.com/generative-ai`) could not be verified during this pass, so it is deliberately **not** listed above.
