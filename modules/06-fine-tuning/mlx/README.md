# MLX lane — fine-tune Mac-native, no CUDA in sight

The PyTorch lane in `../python/` runs on Apple's **MPS** backend — PyTorch's
translation layer onto Metal. It works, but it's a general-purpose framework
squeezing itself onto Apple silicon. **[MLX](https://github.com/ml-explore/mlx)**
is Apple's own array framework, built from scratch for unified memory and the M-series
GPU. `mlx-lm` wraps it with a batteries-included LoRA trainer. On this Mac it
fine-tunes the **same SmolLM2-360M model on the same data several times faster.**

> `mlx-lm` ships **arm64-only** wheels — this lane lives in its own uv project so
> it can't contaminate the cross-platform `../python/` environment.

## Setup

```bash
cd mlx
uv sync                      # installs mlx-lm + datasets (arm64 wheels)
uv run python convert_dataset.py --n 512
```

`convert_dataset.py` pulls `HuggingFaceTB/smoltalk` (everyday-conversations) and
writes `data/train.jsonl` + `data/valid.jsonl` in the **chat schema** mlx-lm reads
natively — one `{"messages": [...]}` object per line. Same data as the PyTorch lab.

## Fine-tune

```bash
uv run python -m mlx_lm lora \
    --model mlx-community/SmolLM2-360M-Instruct \
    --train --data data \
    --iters 100 --batch-size 4 --num-layers 16 \
    --learning-rate 2e-4 --max-seq-length 1024 \
    --steps-per-report 10 \
    --adapter-path adapters
```

This is a short run to feel the speed; bump `--iters` for a real fine-tune.
`--num-layers 16` LoRA-adapts the top 16 transformer blocks (mlx-lm's knob for
"how deep to reach"). The adapter lands in `adapters/adapters.safetensors`.

## Try the adapter

```bash
uv run python -m mlx_lm generate \
    --model mlx-community/SmolLM2-360M-Instruct \
    --adapter-path adapters \
    --prompt "I'm bored this afternoon. Any ideas for something to do at home?"
```

Fuse the adapter into a standalone model for deployment:

```bash
uv run python -m mlx_lm fuse \
    --model mlx-community/SmolLM2-360M-Instruct \
    --adapter-path adapters --save-path fused-model
```

## The teaching point — MLX vs PyTorch/MPS on the same M5

Both lanes fine-tune **SmolLM2-360M on the same everyday-conversations data on
this same Mac.** Measured here (M5, unified memory):

| Lane                        | Throughput      | Peak memory | Notes                         |
|-----------------------------|-----------------|-------------|-------------------------------|
| PyTorch + MPS (`../python`) | **~479 tok/s**  | —           | 200 steps ≈ 12 min            |
| MLX (`mlx_lm.lora`)         | **~3,500 tok/s**| ~2.8 GB     | 100 iters in well under 1 min |

That's roughly a **7× throughput edge** for the Apple-native stack on identical
hardware and model. The framework built *for* the chip beats the framework
*ported to* the chip.

**Honest caveats** — it isn't a perfectly controlled benchmark: MLX ran batch-4
adapting 16 layers while the PyTorch lane ran batch-2 on all-linear targets, and
the two runtimes count tokens slightly differently. The magnitude of the gap,
not the exact multiple, is the lesson: on a Mac, reach for MLX when you can.

## Trade-offs

- **MLX wins:** speed, memory efficiency, dead-simple CLI, native fusion/quantization.
- **PyTorch wins:** portability (the exact same code runs on CUDA in the cloud
  lane), the full `transformers`/`trl`/`peft` ecosystem, and it's what almost
  every model card and tutorial assumes. That's why the course's *primary* lane
  is PyTorch — MLX is the Mac-native power tool you graduate to.
