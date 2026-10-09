# Cloud lane: when your Mac isn't enough

**Local-first is the rule.** Every module in this course runs on your M5 Mac —
that is the design. This page is the *optional* escape hatch: when a fine-tune
is too big, a GPU would save you an hour, or you want to host a demo, here is
how to reach for the cloud without surprises. Skip it entirely if you never need
it.

All prices and limits below were verified in **July 2026** against the Hugging
Face docs cited inline. Check the live pages before spending; they change.

---

## HF Jobs — pay-as-you-go GPUs

Run any command (or uv script) on Hugging Face infrastructure and pay only for
the minutes it runs. **No subscription is required** — you just need a
**positive credit balance** on your account. Billing is **per minute while the
job runs**.

> **The one gotcha: the default timeout is 30 minutes.** A job is killed at 30
> min unless you pass `--timeout`. Always set it for real training runs.

Common flavors and on-demand prices (see the full, current list with
`hf jobs hardware`):

| Flavor | Hardware | Price |
|--------|----------|-------|
| `cpu-basic` | CPU only | $0.01/h |
| `t4-small` | 16 GB NVIDIA T4 | $0.40/h |
| `a10g-small` | 24 GB NVIDIA A10G | $1.00/h |
| `a100-large` | 80 GB NVIDIA A100 | $2.50/h |

List available hardware:

```bash
hf jobs hardware
```

Smoke test — run a container and print a line:

```bash
hf jobs run --name hello --flavor cpu-basic python:3.12 python -c "print('hi')"
```

Run a uv script on a GPU with a real timeout (this is the pattern module 06
uses for fine-tuning):

```bash
hf jobs uv run --name sft --flavor a10g-small --timeout 2h script.py
```

Manage jobs:

```bash
hf jobs ls
```

```bash
hf jobs wait <id>
```

```bash
hf jobs cancel <id>
```

Docs: [Jobs guide](https://huggingface.co/docs/huggingface_hub/guides/jobs) ·
[TRL jobs training](https://huggingface.co/docs/trl/main/en/jobs_training) ·
ready-to-run scripts at [uv-scripts](https://huggingface.co/uv-scripts).

---

## trackio — experiment tracking, free

`trackio` is a free, **wandb-API-compatible** experiment tracker. By default it
runs a **local Gradio dashboard** — no account, no upload. With
`transformers>=4.54`, tracking your training run is one argument in
`TrainingArguments`:

```python
report_to="trackio"
```

Want a shareable, live dashboard? Add a Space id and trackio hosts it for free
on Spaces, backed by an HF Dataset:

```python
trackio_space_id="username/space"
```

Docs: <https://huggingface.co/docs/trackio>.

---

## Spaces — host your demos, free

The course produces demos (the module 03 Gradio tokenizer playground, for one).
Host them on [Spaces](https://huggingface.co/spaces):

- **Free CPU tier:** 2 vCPU / 16 GB RAM — enough for the Gradio apps in this
  course.
- **ZeroGPU:** a free, *shared* GPU for demos.
  - **Free account:** 5 min of GPU per day, up to **2** hostable ZeroGPU Spaces.
  - **PRO ($9/mo):** 40 min of GPU per day, up to **10** Spaces, highest queue
    priority.

---

## Inference Providers — free monthly credits

Calling hosted models through
[Inference Providers](https://huggingface.co/docs/inference-providers) comes
with free monthly credits: **$0.10/mo** on a free account, **$2/mo** with PRO.

---

## Colab — zero-cost training fallback

Google Colab's **free T4** tier is a fine no-cost place to run a training
notebook when you don't want to spend Jobs credits. Its limits (session length,
GPU availability) are unpublished and vary, so treat it as best-effort.

---

## Recommended path

> **Start at $0.** Local M5 for everything, Colab's free T4 when you want a GPU,
> free Spaces to host a demo. This covers the whole course.
>
> **Want cloud GPU runs?** Load about **$5** of credits and use HF Jobs — a full
> fine-tune on `t4-small` costs **cents**. Always pass `--timeout`.
>
> **Only go PRO ($9/mo)** if you specifically want ZeroGPU demos with real daily
> quota and priority. Nothing in the course requires it.
