# Resources — the Hugging Face Learn catalog

This course was distilled from the free, open [Hugging Face learning
platform](https://huggingface.co/learn). We took the parts that teach you to
**build** — autograd, attention, tokenizers, fine-tuning — and rebuilt them from
scratch, twice, with pictures. But HF Learn is much bigger than what we cover,
and every one of our modules has a natural sequel there.

This page is the full menu. It maps each of our modules back to the HF course
that goes deeper, then catalogs every course on the platform — what it covers,
what you need first, how hands-on it is, and whether it grants a certificate.
When you finish a module here and want more, this is where you go.

---

## Go deeper, module by module

Each row: what you built here → where to continue on HF Learn.

| Our module | Continue with | Why |
|---|---|---|
| [01 · autograd](modules/01-autograd) · [02 · neural-networks](modules/02-neural-networks) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 1–2 · [CV Course](https://huggingface.co/learn/computer-vision-course) unit 1 | Foundations background: what a Transformer is, and the fundamentals-of-vision framing for the networks you just hand-wrote. |
| [03 · tokenization](modules/03-tokenization) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 6 | The full tokenizers library — training a tokenizer, fast tokenizers, and the pipeline behind the BPE you built by hand. |
| [04 · attention-transformer](modules/04-attention-transformer) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 1 ("how Transformers work") | The conceptual companion to your char-GPT: attention, architectures, and the encoder/decoder families. |
| [05 · transformers-library](modules/05-transformers-library) | [LLM Course](https://huggingface.co/learn/llm-course) ch. 1–4 | `pipeline`, `AutoModel`, fine-tuning basics — the canonical tour of the library whose anatomy you dissected. |
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
