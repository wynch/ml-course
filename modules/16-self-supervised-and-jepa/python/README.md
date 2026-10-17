# mlcourse-16-self-supervised-jepa (Python lane)

Python code for module 16, self-supervised learning and JEPA. See the module
overview one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), CIFAR-10 loader, results I/O
- `src/ssl_core.py`: the shared parts: the encoder, augmentations, masks, losses, anti-collapse terms, collapse meters, probes
- `src/pretrain.py`: sections 1 and 3, one objective per run (random, supervised, simclr, mae, jepa), then the probes
- `src/collapse.py`: section 2, the JEPA objective with one of five guards against collapse
- `src/dinov2_probe.py`: section 4, a linear probe on frozen DINOv2-small features
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against `ssl_core.py`; the results against the budget, each other and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
