# Performance sidebar — why frameworks sit on optimized kernels

The forward pass of our `784 -> 256` MLP is, at its core, one matrix multiply:
`X(256, 784) @ W(784, 256)` for a batch of 256. That single operation is where
essentially all the FLOPs go. This sidebar measures the same multiply three
ways in Zig and compares against numpy's BLAS, to make one point concrete:

> The algorithm ("multiply two matrices") is fixed. The performance is almost
> entirely about *memory access order* and *hand-tuned kernels* — which is
> exactly what PyTorch/TensorFlow/JAX buy you for free.

## The three Zig variants (`src/main.zig`)

| variant           | what changed                                                        |
| ----------------- | ------------------------------------------------------------------- |
| `naive (ijk)`     | textbook triple loop; inner loop strides **down a column** of `B`   |
| `reordered (ikj)` | swap the two inner loops so the inner loop walks `B`/`C` **rows**    |
| `blocked (tiled)` | process 64×64 tiles so operands stay hot in L1/L2 across reuse       |

All three compute the identical result (the benchmark verifies `max|diff| = 0`
against the reordered reference). Only the memory access pattern differs.

## Measured results

Apple M5, Zig 0.16.0 `ReleaseFast`, single thread, best of 20 runs. Shapes:
`C(256×256) = A(256×784) · B(784×256)`, `2·M·N·K ≈ 1.03×10^8` FLOPs.

| variant                        | time (ms) | GFLOP/s | speedup |
| ------------------------------ | --------: | ------: | ------: |
| naive (ijk)                    |    31.06  |    3.31 |  1.00x  |
| reordered (ikj)                |    13.79  |    7.45 |  2.25x  |
| blocked (tiled)                |    12.88  |    7.98 |  2.41x  |
| **numpy `@` (Accelerate BLAS)**|   **0.058** | **1773** | **~740x** |

(Your absolute numbers will vary; the *ratios* are the lesson. Reproduce with
`zig build run` here and `uv run python bench_numpy.py` in `../python`.)

## What the numbers say

* **Loop order alone is a 2.25x win.** The naive `ijk` inner loop reads
  `B[k*N + j]` — a jump of `N` floats (a whole cache line, mostly wasted) every
  iteration. The `ikj` order reads `B` and writes `C` contiguously, so each
  cache line is fully used and the compiler can auto-vectorize the inner loop.
  Same FLOPs, same result, ~2x faster — purely from respecting the cache.

* **Blocking adds a little more** by keeping tiles resident across reuse. On
  these modest sizes the working set is already small, so the gain over `ikj`
  is minor; it grows with larger matrices.

* **numpy is another ~200–700x beyond our best loop.** Apple's Accelerate BLAS
  (`sgemm`) uses register blocking, hand-written SIMD/AMX microkernels, prefetch,
  and multithreading — decades of kernel engineering. Our tidy Zig loop cannot
  touch it, and neither could a tidy C or Rust loop.

## The takeaway for the course

This is *why* deep-learning frameworks exist as a thin Python layer over big
compiled kernels: `numpy`, BLAS (Accelerate/OpenBLAS/MKL), cuBLAS, cuDNN. When
you write `x @ W` in numpy — or `nn.Linear` in PyTorch — you are not running a
Python loop; you are dispatching to a kernel like the one that beat our Zig by
two orders of magnitude. Writing correct backprop from scratch (module 02) and
letting an optimized kernel run it fast are two separate concerns, and real
systems keep them separate on purpose.

## Reproduce

```bash
zig build run
```

```bash
cd ../python && uv run python bench_numpy.py
```
