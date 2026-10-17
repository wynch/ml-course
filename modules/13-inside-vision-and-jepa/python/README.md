# mlcourse-13-inside-vision-jepa (Python lane)

Python code for module 13, inside vision and JEPA models. See the module
overview one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), paths, results I/O
- `src/fetch_imagenette.py`: downloads imagenette2-160 once and checks its SHA-256
- `src/v_neuron.py`: section 1, InceptionV1 channels 45 to 65 of layer 4e on six image sets, trained and random weights
- `src/feature_vis.py`: section 2, Lucent renders of 4e:54, 4e:55 and 4e:56
- `src/clip_inside.py`: section 3, a class probe per CLIP block and three CLIP MLP neurons
- `src/grid_encoder.py`: the gridjepa encoder class, its input builder and the probe helpers
- `src/jepa_probe.py`: section 4, position, neighbour and path-length probes on the frozen encoder
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: the solutions against the scripts; the results against the budget, the earlier measurement and the explorable

## Quickstart

```bash
uv sync
uv run python src/fetch_imagenette.py
uv run pytest
```
