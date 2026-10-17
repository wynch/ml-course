"""Section 1: one InceptionV1 channel, 4e:55, on six sets of 200 images.

The claim under test: channel 55 of layer mixed4e responds to V shapes more
than to cats and cars. We draw 200 Vs and 200 Os, take 200 cats, 200 CIFAR
cars (32 px, upscaled), 200 garbage trucks and 200 other imagenette photos,
and record each image's strongest response of channels 45 to 65 (before the
ReLU, max over the 14 x 14 map). Then the same with random weights.

    uv run python src/v_neuron.py --figures
"""

from __future__ import annotations

import argparse
import glob
import io
import math
import os

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch
from PIL import Image, ImageDraw

from common import IMAGENETTE, Guard, free, get_device, need_imagenette, write_results

TRUCK = "n03417042"
TARGET = 55
CHANNELS = list(range(45, 66))
N = 200
SIZE = 224
SETS = ["V", "O", "cat", "car", "truck", "imagenette"]
LUCENT_WEIGHTS = "inception5h.pth, 27.0 MB, sha256 44f1f1e0734fa77edd0ab948c0d4d5c9ab6f4efce10b02d793fd14ab2d71a73b"


# ---------------------------------------------------------------- drawing
def _colours(rng: np.random.Generator) -> tuple[tuple, tuple]:
    """Random background and stroke colours, luma apart by at least 0.3."""
    while True:
        bg = rng.random(3)
        fg = rng.random(3)
        luma = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        if abs(luma(bg) - luma(fg)) >= 0.3:
            return tuple(int(255 * v) for v in bg), tuple(int(255 * v) for v in fg)


def draw_v(rng: np.random.Generator) -> tuple[Image.Image, dict]:
    """An upright V (vertex down), drawn at 2x and downsampled for anti-aliasing."""
    S = 2 * SIZE
    bg, fg = _colours(rng)
    opening = rng.uniform(25, 90)            # degrees between the two arms
    arm = rng.uniform(50, 100) * 2           # px at 2x
    width = int(round(rng.uniform(3, 12) * 2))
    tilt = rng.uniform(-20, 20)              # degrees off upright
    half = math.radians(opening / 2)
    ends = []
    for sign in (-1, 1):
        a = math.radians(-90 + tilt) + sign * half   # up is -y
        ends.append((arm * math.cos(a), arm * math.sin(a)))
    xs = [0, ends[0][0], ends[1][0]]
    ys = [0, ends[0][1], ends[1][1]]
    m = width
    vx = rng.uniform(m - min(xs), S - m - max(xs))
    vy = rng.uniform(m - min(ys), S - m - max(ys))
    img = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(img)
    for ex, ey in ends:
        d.line([(vx, vy), (vx + ex, vy + ey)], fill=fg, width=width)
    r = width / 2
    d.ellipse([vx - r, vy - r, vx + r, vy + r], fill=fg)   # round the vertex
    params = {"opening": round(opening, 2), "arm": round(arm / 2, 2), "width": width / 2,
              "tilt": round(tilt, 2), "x": round(vx / 2, 1), "y": round(vy / 2, 1)}
    return img.resize((SIZE, SIZE), Image.LANCZOS), params


def draw_o(rng: np.random.Generator) -> tuple[Image.Image, dict]:
    S = 2 * SIZE
    bg, fg = _colours(rng)
    r = rng.uniform(25, 60) * 2
    aspect = rng.uniform(0.8, 1.25)
    rx, ry = r * aspect, r / aspect
    width = int(round(rng.uniform(3, 12) * 2))
    m = width
    cx = rng.uniform(rx + m, S - rx - m)
    cy = rng.uniform(ry + m, S - ry - m)
    img = Image.new("RGB", (S, S), bg)
    ImageDraw.Draw(img).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=fg, width=width)
    params = {"rx": round(rx / 2, 2), "ry": round(ry / 2, 2), "width": width / 2,
              "x": round(cx / 2, 1), "y": round(cy / 2, 1)}
    return img.resize((SIZE, SIZE), Image.LANCZOS), params


# ---------------------------------------------------------------- photos
def square(img: Image.Image) -> Image.Image:
    """Shorter side to 224 (bilinear), then a centre crop."""
    img = img.convert("RGB")
    w, h = img.size
    s = SIZE / min(w, h)
    img = img.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return img.crop((l, t, l + SIZE, t + SIZE))


# Test splits only, pinned to the revisions the committed numbers came from.
CATS = ("Bingsu/Cat_and_Dog", "data/test-00000-of-00001.parquet",
        "77396f1031920d1c116ec6d63ac397ff6aa492d3")              # 45.8 MB
CARS = ("uoft-cs/cifar10", "plain_text/test-00000-of-00001.parquet",
        "0b2714987fa478483af9968de7c934580d0bb9a2")              # 23.9 MB


def parquet_images(source: tuple, img_col: str, label_col: str, label: int, seed: int):
    """200 images of one label from a Hub parquet file (downloaded once, then cached)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, filename, revision = source
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    t = pq.read_table(path)
    labels = np.array(t.column(label_col).to_pylist())
    idx = np.flatnonzero(labels == label)
    pick = np.sort(np.random.default_rng(seed).choice(idx, size=N, replace=False))
    col = t.column(img_col)
    out = [Image.open(io.BytesIO(col[int(i)].as_py()["bytes"])).convert("RGB") for i in pick]
    return out, [int(i) for i in pick]


def imagenette_files(classes: list[str], seed: int) -> list[str]:
    files = []
    for c in classes:
        files += sorted(glob.glob(str(IMAGENETTE / "val" / c / "*.JPEG")))
    pick = np.sort(np.random.default_rng(seed).choice(len(files), size=N, replace=False))
    return [files[i] for i in pick]


def build_sets() -> tuple[dict, dict]:
    """Six lists of 200 PIL images, 224 x 224, and what each image is."""
    need_imagenette()
    sets, meta = {}, {}
    rng = np.random.default_rng(0)
    vs = [draw_v(rng) for _ in range(N)]
    sets["V"], meta["V"] = [im for im, _ in vs], [p for _, p in vs]
    rng = np.random.default_rng(0)
    os_ = [draw_o(rng) for _ in range(N)]
    sets["O"], meta["O"] = [im for im, _ in os_], [p for _, p in os_]

    cats, idx = parquet_images(CATS, "image", "labels", 0, seed=0)
    sets["cat"], meta["cat"] = [square(im) for im in cats], idx
    cars, idx = parquet_images(CARS, "img", "label", 1, seed=0)
    sets["car"], meta["car"] = [im.resize((SIZE, SIZE), Image.BILINEAR) for im in cars], idx
    trucks = imagenette_files([TRUCK], seed=0)
    sets["truck"] = [square(Image.open(f)) for f in trucks]
    meta["truck"] = [os.path.basename(f) for f in trucks]
    others = sorted(os.path.basename(p) for p in glob.glob(str(IMAGENETTE / "val" / "n*")))
    others = [c for c in others if c != TRUCK]
    files = imagenette_files(others, seed=0)
    sets["imagenette"] = [square(Image.open(f)) for f in files]
    meta["imagenette"] = [f.split("/val/")[1] for f in files]
    return sets, meta


def to_tensor(imgs: list[Image.Image]) -> torch.Tensor:
    a = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs])
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------- model
def load_inception(pretrained: bool, device: torch.device):
    """Lucent's InceptionV1 (the TensorFlow 'inception5h' weights).

    ``pretrained=False`` keeps PyTorch's default initialisation after
    ``torch.manual_seed(0)``: the random-weight control.
    """
    from lucent.modelzoo import inceptionv1

    if not pretrained:
        torch.manual_seed(0)
    return inceptionv1(pretrained=pretrained).to(device).eval()


@torch.no_grad()
def channel_max(model, x: torch.Tensor, device: torch.device, batch: int = 50) -> np.ndarray:
    """(N, 3, 224, 224) in [0, 1] -> (N, 21): spatial max of channels 45..65, pre-ReLU.

    Channels 0-255 of mixed4e are its 1x1 branch, so channel c of mixed4e
    before the ReLU is channel c of ``mixed4e_1x1_pre_relu_conv``.
    """
    seen = {}
    h = model.mixed4e_1x1_pre_relu_conv.register_forward_hook(
        lambda mod, inp, out: seen.__setitem__("a", out))
    out = []
    try:
        for i in range(0, len(x), batch):
            xb = (x[i:i + batch] * 255 - 117).to(device)     # Lucent's InceptionV1 input range
            model(xb)
            out.append(seen["a"][:, CHANNELS].amax(dim=(2, 3)).float().cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(out)


def gaps(acts: dict, vehicle: str = "car") -> np.ndarray:
    """Per channel: (median V - max(median cat, median vehicle)) / pooled SD."""
    med = {k: np.median(acts[k], axis=0) for k in ("V", "cat", vehicle)}
    var = np.mean([acts[k].var(axis=0, ddof=1) for k in ("V", "cat", vehicle)], axis=0)
    return (med["V"] - np.maximum(med["cat"], med[vehicle])) / np.sqrt(var)


def rank_of(g: np.ndarray, ch: int) -> int:
    order = np.argsort(-g, kind="stable")
    return int(np.flatnonzero(np.array(CHANNELS)[order] == ch)[0]) + 1


def summarise(acts: dict) -> dict:
    j = CHANNELS.index(TARGET)
    g = gaps(acts, "car")
    gt = gaps(acts, "truck")
    return {
        "median": {k: round(float(np.median(acts[k][:, j])), 4) for k in SETS},
        "q25": {k: round(float(np.quantile(acts[k][:, j], 0.25)), 4) for k in SETS},
        "q75": {k: round(float(np.quantile(acts[k][:, j], 0.75)), 4) for k in SETS},
        "gap_by_channel": {str(c): round(float(v), 4) for c, v in zip(CHANNELS, g)},
        "gap_trucks_by_channel": {str(c): round(float(v), 4) for c, v in zip(CHANNELS, gt)},
        "gap_55": round(float(g[j]), 4),
        "rank_55": rank_of(g, TARGET),
        "gap_trucks_55": round(float(gt[j]), 4),
        "rank_trucks_55": rank_of(gt, TARGET),
        "top_channel": CHANNELS[int(np.argmax(g))],
        "top_channel_trucks": CHANNELS[int(np.argmax(gt))],
    }


def thumbs(sets: dict, acts: dict, k: int = 5, px: int = 56) -> dict:
    """The k highest and k lowest images of each set for 4e:55, as small JPEGs (base64)."""
    import base64

    j = CHANNELS.index(TARGET)
    out = {}
    for s in SETS:
        order = np.argsort(-acts[s][:, j], kind="stable")
        picks = [int(i) for i in list(order[:k]) + list(order[-k:])]
        row = []
        for i in picks:
            buf = io.BytesIO()
            sets[s][i].resize((px, px), Image.BILINEAR).save(buf, "JPEG", quality=72)
            row.append({"i": i, "a": round(float(acts[s][i, j]), 2),
                        "jpg": base64.b64encode(buf.getvalue()).decode()})
        out[s] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("v_neuron")
    device = get_device()

    sets, meta = build_sets()
    x = {k: to_tensor(v) for k, v in sets.items()}
    guard.tick("build_sets")

    model = load_inception(True, device)
    acts = {k: channel_max(model, x[k], device) for k in SETS}
    guard.tick("trained")
    cpu = load_inception(True, torch.device("cpu"))
    a_cpu = channel_max(cpu, x["V"][:8], torch.device("cpu"))
    mps_vs_cpu = float(np.abs(a_cpu - acts["V"][:8]).max())
    del cpu, model
    free()
    guard.tick("cpu_check")
    rmodel = load_inception(False, device)
    racts = {k: channel_max(rmodel, x[k], device) for k in SETS}
    del rmodel
    free()
    guard.tick("random")

    tr, rd = summarise(acts), summarise(racts)
    j = CHANNELS.index(TARGET)
    payload = {
        "model": f"Lucent InceptionV1 ({LUCENT_WEIGHTS})",
        "unit": "mixed4e_1x1_pre_relu_conv channel c, spatial max over 14 x 14, before ReLU",
        "channels": CHANNELS,
        "target": TARGET,
        "n_per_set": N,
        "sets": SETS,
        "trained": tr,
        "random": rd,
        "mps_vs_cpu_max_abs": mps_vs_cpu,
        "activation_range_55": [round(float(min(acts[k][:, j].min() for k in SETS)), 4),
                                round(float(max(acts[k][:, j].max() for k in SETS)), 4)],
        "device": device.type,
        "guard": guard.report(),
    }
    write_results("v_neuron", payload)
    # Every activation, rounded, for the explorable's strips and exercise (b).
    write_results("v_neuron_acts", {
        "channels": CHANNELS, "sets": SETS,
        "trained": {k: np.round(acts[k].T.astype(np.float64), 1).tolist() for k in SETS},
        "random_55": {k: np.round(racts[k][:, j].astype(np.float64), 4).tolist() for k in SETS},
        "thumbs": thumbs(sets, acts),
        "image_ids": meta,
    }, compact=True)
    print(f"4e:55 medians: {tr['median']}")
    print(f"gap {tr['gap_55']} SD, rank {tr['rank_55']} of 21; random-weight gap {rd['gap_55']}; "
          f"MPS vs CPU {mps_vs_cpu:.2e}")
    if args.figures:
        import figures

        figures.fig_v_neuron()
        figures.fig_v_examples()


if __name__ == "__main__":
    main()
