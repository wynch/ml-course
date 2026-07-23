"""Time numpy's BLAS-backed matmul on the module's actual layer sizes.

This is the counterpart to the Zig benchmark in ../zig. Same shapes:
C(256x256) = A(256x784) @ B(784x256), i.e. one forward matmul of the 784->256
layer for a batch of 256. Run:

    uv run python bench_numpy.py

numpy dispatches to an optimized BLAS (Apple Accelerate on this Mac), so this
is the "framework kernel" number to compare our hand-written loops against.
"""

from __future__ import annotations

import time

import numpy as np

M, K, N = 256, 784, 256


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)

    C = A @ B  # warm up
    best = float("inf")
    for _ in range(50):
        t = time.perf_counter()
        C = A @ B
        best = min(best, time.perf_counter() - t)

    flops = 2 * M * N * K
    print(f"numpy float32 matmul  C({M}x{N}) = A({M}x{K}) @ B({K}x{N})")
    print(f"best of 50 runs: {best * 1e3:.4f} ms   {flops / best / 1e9:.1f} GFLOP/s")
    blas = np.__config__.show(mode="dicts")["Build Dependencies"]["blas"]["name"]
    print(f"BLAS backend: {blas}")
    _ = C  # keep result alive


if __name__ == "__main__":
    main()
