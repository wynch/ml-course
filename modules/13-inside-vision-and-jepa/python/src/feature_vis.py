"""Section 2: feature visualisation. Lucent paints the image that drives a channel.

Start from noise (in a decorrelated Fourier basis), then follow the gradient
of one channel's mean activation with respect to the pixels, under small
random jitters, scales and rotations. We render 4e:55 and its two neighbours,
54 and 56, and record the activation each render reaches against the best real
image of section 1.

Lucent 0.1.8 picks its device as "CUDA or CPU", so this runs on the CPU.

    uv run python src/feature_vis.py --figures
"""

from __future__ import annotations

import argparse
import base64
import io
import random

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch
from PIL import Image

from common import Guard, read_results, write_results
from v_neuron import load_inception

CHANNELS = [54, 55, 56]
STEPS = 512
SIZE = 128


def render(model, channel: int, seed: int = 0) -> np.ndarray:
    """One render of mixed4e:channel, (SIZE, SIZE, 3) floats in [0, 1]."""
    from lucent.optvis import param, render as lrender

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    imgs = lrender.render_vis(
        model, f"mixed4e:{channel}",
        param_f=lambda: param.image(SIZE, fft=True, decorrelate=True),
        thresholds=(STEPS,), show_image=False, progress=False,
    )
    return imgs[-1][0]


@torch.no_grad()
def spatial_max(model, img: np.ndarray, channel: int) -> float:
    """Pre-ReLU spatial max of mixed4e:channel on one image, as in section 1."""
    seen = {}
    h = model.mixed4e_1x1_pre_relu_conv.register_forward_hook(
        lambda m, i, o: seen.__setitem__("a", o))
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
    model(x * 255 - 117)
    h.remove()
    return float(seen["a"][0, channel].amax())


def to_jpeg(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("feature_vis")
    torch.set_num_threads(4)
    model = load_inception(True, torch.device("cpu"))
    v = read_results("v_neuron_acts")
    renders = []
    for c in CHANNELS:
        img = render(model, c)
        a = spatial_max(model, img, c)
        best = max(max(v["trained"][s][v["channels"].index(c)]) for s in v["sets"])
        renders.append({"channel": c, "activation": round(a, 1),
                        "best_real_image": round(float(best), 1),
                        "ratio_to_best_real": round(a / best, 2), "jpg": to_jpeg(img)})
        guard.tick(f"render_{c}")
        print(f"4e:{c}: render reaches {a:.1f}; best real image {best:.1f}")
    write_results("feature_vis", {
        "model": "Lucent InceptionV1", "objective": "mean of mixed4e:c after ReLU",
        "steps": STEPS, "size": SIZE, "param": "Fourier basis, decorrelated colour",
        "transforms": "Lucent standard: pad 12, jitter 8, scale 0.9-1.1, rotate -10..10 degrees",
        "seed": 0, "device": "cpu", "renders": renders, "guard": guard.report(),
    })
    if args.figures:
        import figures

        figures.fig_feature_vis()


if __name__ == "__main__":
    main()
