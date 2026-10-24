#!/bin/sh
# Every optimiser run, chunk by chunk, until each prints "remaining": 0.
cd "$(dirname "$0")"
for seed in 0 1 2; do
  for method in ape opro textgrad gepa; do
    until [ -f "../results/runs/${method}_s${seed}.json" ]; do
      uv run python src/optimise.py "$method" "$seed" || exit 1
    done
  done
done
