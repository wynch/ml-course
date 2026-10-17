# mlcourse-15-geometry-of-representations (Python lane)

Python code for module 15, geometry of representations. See the module
overview one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), data loaders, module 02's MLP, results I/O
- `src/idest.py`: the two intrinsic-dimension estimators (TwoNN, MLE) and the curved cubes of known dimension
- `src/regions.py`: section 1, exact linear regions of a 2-D net, and regions along lines against width
- `src/intrinsic_dim.py`: section 2, intrinsic dimension of known sets, FashionMNIST through module 02's MLP, and ViT-base on imagenette
- `src/manifold.py`: section 3, PCA against intrinsic dimension, and midpoints of two images
- `src/cka.py`: section 4, linear CKA between two seeds of module 02's MLP, trained and at init
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the estimator check on sets of known dimension; the solutions against the scripts; the results against the budget and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
