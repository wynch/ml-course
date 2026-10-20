# Cloud lane — run the same SFT on a rented GPU (optional, costs money)

The local lab fits a 360M model on your Mac's GPU. When you want to fine-tune
something bigger (SmolLM3-3B) on more data than MPS can chew through in a coffee
break, you rent a GPU by the second with **Hugging Face Jobs**. `cloud/sft_job.py`
is the local recipe, packaged as a self-contained [PEP 723](https://peps.python.org/pep-0723/)
uv script so it runs with **no repo checkout** — the inline dependency block at
the top of the file is all Jobs needs.

> ⚠️ **This lane is optional and it is not free.** HF Jobs requires **prepaid
> credits** (a Pro/Team/Enterprise plan or pay-as-you-go). Nothing here runs on
> your machine, and this course never launches it for you. The commands below
> are yours to run if and when you choose to.

## What it costs

Jobs bills the GPU **flavor** by the second. Ballpark on-demand rates:

| Flavor       | GPU          | ~$/hour | Good for                                  |
|--------------|--------------|---------|-------------------------------------------|
| `a10g-small` | 1× A10G 24GB | ~$1.00  | 3B LoRA in bf16, the default here         |
| `l4x1`       | 1× L4 24GB   | ~$0.80  | same job, also bf16                       |
| `t4-small`   | 1× T4 16GB   | ~$0.40  | the 360M smoke test; no bf16, fp16 only   |

We have not run this job, so we quote no run time. Measure it: run the timing
test below on the flavor you chose, read the seconds per step from the
logs, multiply by `MAX_STEPS`, and set `--timeout` to twice that. Cost is that
time times the hourly rate. Always check current pricing — rates change.

The default run is SmolLM3-3B for 90 steps on the whole `everyday-conversations`
subset (2,260 conversations, packed into 1,024-token sequences, 16 per step):
about three passes, like the local lab.

## Launch it

From `modules/06-fine-tuning/`:

```bash
hf jobs uv run \
    --name sft-smollm \
    --flavor a10g-small \
    --timeout 2h \
    --secrets HF_TOKEN \
    cloud/sft_job.py
```

- Set `--timeout` from the timing test below; `2h` is a placeholder.
- `--secrets HF_TOKEN` forwards your token into the job so it can read gated
  models and push the adapter back to the Hub.

### ⚠️ The 30-minute timeout trap

If you omit `--timeout`, Jobs applies a **default 30-minute** limit and your run
is **killed mid-training** with a confusing non-error exit. Any real fine-tune
needs an explicit, generous `--timeout` (twice the measured time). This is the single
most common way a first cloud run "silently fails".

## Watch and manage the run

```bash
hf jobs ls                 # list your jobs and their status
hf jobs logs sft-smollm    # stream stdout/stderr
hf jobs logs -f sft-smollm # follow (tail -f style)
hf jobs wait sft-smollm    # block until it finishes (handy in scripts)
hf jobs cancel sft-smollm  # stop it (and stop paying)
```

## Live loss curves with trackio on a free Space

The local lab logs trackio to a local SQLite db. For a **cloud** run you want to
watch the loss from your laptop while the GPU works. Set a Space id and trackio
will create/host a **free** monitoring Space:

```bash
hf jobs uv run --name sft-smollm --flavor a10g-small --timeout 2h \
    --secrets HF_TOKEN \
    -e TRACKIO_SPACE=your-username/smollm-sft \
    -e PUSH_REPO=your-username/smollm-sft-lora \
    cloud/sft_job.py
```

`sft_job.py` reads `TRACKIO_SPACE` and passes it as `trackio_space_id`, so metrics
stream to `https://huggingface.co/spaces/your-username/smollm-sft` in real time.
`PUSH_REPO` makes the job upload the finished LoRA adapter to your Hub repo.

## Retargeting without editing the file

Every knob is an env var (`-e KEY=VALUE`):

| Env var        | Default                              | Meaning                        |
|----------------|--------------------------------------|--------------------------------|
| `MODEL_ID`     | `HuggingFaceTB/SmolLM3-3B`           | base model to fine-tune        |
| `MAX_STEPS`    | `90`                                 | training steps                 |
| `TRAIN_SLICE`  | `2260`                               | # examples (the whole subset)  |
| `PUSH_REPO`    | `""` (no push)                       | Hub repo for the adapter       |
| `TRACKIO_SPACE`| `""` (no live logging)              | Space id for live monitoring   |

Two cheap tests before the full run. First the pipeline, on the small model
and the cheapest GPU:

```bash
hf jobs uv run --name sft-smoke --flavor t4-small --timeout 30m \
    --secrets HF_TOKEN \
    -e MODEL_ID=HuggingFaceTB/SmolLM2-360M-Instruct -e MAX_STEPS=50 \
    cloud/sft_job.py
```

Then the timing: the real model on the flavor you chose, for a few steps.
Its seconds per step, times `MAX_STEPS`, times two, is your `--timeout`:

```bash
hf jobs uv run --name sft-timing --flavor a10g-small --timeout 30m \
    --secrets HF_TOKEN -e MAX_STEPS=10 \
    cloud/sft_job.py
```

## Further reading

- TRL on Jobs (official guide): <https://huggingface.co/docs/trl/main/en/jobs_training>
