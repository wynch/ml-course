"""Section 3: inside CLIP ViT-B/32's image tower, on imagenette.

(a) A class probe per layer: logistic regression on the CLS vector after each
    of the 12 blocks (and the embeddings), 10 imagenette classes, trained
    weights against the same architecture with random weights.
(b) Three MLP neurons drawn at random (layers 3, 7 and 11), each read by the
    8 images that drive it hardest (max over the 49 patch tokens).

    uv run python src/clip_inside.py --figures
"""

from __future__ import annotations

import argparse
import base64
import glob
import io

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch
from PIL import Image

from common import (CLIP_ID, CLIP_REVISION, IMAGENETTE, IMAGENETTE_CLASSES, Guard, free,
                    get_device, need_imagenette, write_results)

N_TRAIN, N_TEST = 100, 50          # per class
NEURON_LAYERS = [3, 7, 11]         # counted from 0, as in encoder.layers[i]
WIDTH = 3072                       # MLP hidden size of ViT-B/32
TOP = 8


def pick_images() -> tuple[list[str], np.ndarray, np.ndarray]:
    """1,000 training images (100 a class) and 500 held-out ones (50 a class)."""
    need_imagenette()
    rng = np.random.default_rng(0)
    files, labels, is_test = [], [], []
    for c, wnid in enumerate(sorted(IMAGENETTE_CLASSES)):
        for split, n, flag in (("train", N_TRAIN, False), ("val", N_TEST, True)):
            pool = sorted(glob.glob(str(IMAGENETTE / split / wnid / "*.JPEG")))
            for i in np.sort(rng.choice(len(pool), size=n, replace=False)):
                files.append(pool[i]); labels.append(c); is_test.append(flag)
    return files, np.array(labels), np.array(is_test)


def load_clip(random_weights: bool, device: torch.device):
    from transformers import CLIPVisionConfig, CLIPVisionModel

    if random_weights:
        cfg = CLIPVisionConfig.from_pretrained(CLIP_ID, revision=CLIP_REVISION)
        torch.manual_seed(0)
        model = CLIPVisionModel(cfg)
    else:
        model = CLIPVisionModel.from_pretrained(CLIP_ID, revision=CLIP_REVISION)
    return model.to(device).eval()


def pixels(files: list[str]) -> torch.Tensor:
    from transformers import CLIPImageProcessor

    proc = CLIPImageProcessor.from_pretrained(CLIP_ID, revision=CLIP_REVISION)
    out = []
    for i in range(0, len(files), 100):
        imgs = [Image.open(f).convert("RGB") for f in files[i:i + 100]]
        out.append(proc(images=imgs, return_tensors="pt")["pixel_values"])
    return torch.cat(out)


@torch.no_grad()
def run(model, px: torch.Tensor, device: torch.device, neurons: list[tuple[int, int]] | None,
        batch: int = 100):
    """CLS vector after every block (N, 13, 768) and, if asked, each neuron's patch max."""
    layers = getattr(model, "vision_model", model).encoder.layers     # transformers 5 flattened it
    seen, handles = {}, []
    if neurons:
        for L, _ in neurons:
            handles.append(layers[L].mlp.fc2.register_forward_pre_hook(
                lambda m, inp, L=L: seen.__setitem__(L, inp[0])))
    cls, nmax = [], []
    try:
        for i in range(0, len(px), batch):
            out = model(pixel_values=px[i:i + batch].to(device), output_hidden_states=True)
            cls.append(torch.stack([h[:, 0] for h in out.hidden_states], 1).float().cpu())
            if neurons:
                # Token 0 is CLS; tokens 1..49 are the 7 x 7 patches.
                nmax.append(torch.stack([seen[L][:, 1:, n].amax(1) for L, n in neurons], 1).float().cpu())
    finally:
        for h in handles:
            h.remove()
    return torch.cat(cls).numpy(), (torch.cat(nmax).numpy() if neurons else None)


def probe_by_layer(cls: np.ndarray, y: np.ndarray, test: np.ndarray) -> list[float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    acc = []
    for L in range(cls.shape[1]):
        sc = StandardScaler().fit(cls[~test, L])
        clf = LogisticRegression(C=0.1, max_iter=3000)
        clf.fit(sc.transform(cls[~test, L]), y[~test])
        acc.append(round(float(clf.score(sc.transform(cls[test, L]), y[test])), 4))
    return acc


def tile(path: str, px: int = 96) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((px, px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("clip_inside")
    device = get_device()
    names = [IMAGENETTE_CLASSES[w] for w in sorted(IMAGENETTE_CLASSES)]

    files, y, test = pick_images()
    px = pixels(files)
    guard.tick("load_images")

    rng = np.random.default_rng(0)
    neurons = [(L, int(rng.integers(WIDTH))) for L in NEURON_LAYERS]
    model = load_clip(False, device)
    cls, nmax = run(model, px, device, neurons)
    guard.tick("trained")
    cpu = load_clip(False, torch.device("cpu"))
    cls_cpu, _ = run(cpu, px[:16], torch.device("cpu"), None)
    mps_vs_cpu = float(np.abs(cls_cpu[:, -1] - cls[:16, -1]).max())
    cls_span = [float(cls[:16, -1].min()), float(cls[:16, -1].max())]
    del model, cpu
    free()
    rmodel = load_clip(True, device)
    rcls, _ = run(rmodel, px, device, None)
    del rmodel
    free()
    guard.tick("random")

    acc_tr = probe_by_layer(cls, y, test)
    acc_rd = probe_by_layer(rcls, y, test)
    guard.tick("probes")

    neuron_out = []
    for k, (L, n) in enumerate(neurons):
        order = np.argsort(-nmax[:, k], kind="stable")[:TOP]
        top_classes = [names[y[i]] for i in order]
        counts = {c: top_classes.count(c) for c in dict.fromkeys(top_classes)}
        neuron_out.append({
            "id": f"L{L}/N{n}", "layer": L, "neuron": n,
            "max": round(float(nmax[order[0], k]), 3),
            "median_all": round(float(np.median(nmax[:, k])), 3),
            "top": [{"file": files[i].split("/imagenette2-160/")[1], "class": names[y[i]],
                     "activation": round(float(nmax[i, k]), 3), "jpg": tile(files[i])} for i in order],
            "top_class_counts": counts,
            "most_common_share": round(max(counts.values()) / TOP, 3),
        })
        print(f"L{L}/N{n}: top-{TOP} classes {counts}")

    best = int(np.argmax(acc_tr))
    write_results("clip", {
        "model": f"{CLIP_ID} @ {CLIP_REVISION[:7]}, vision tower only",
        "data": "imagenette2-160 (fast.ai), shorter side 160 px, CLIP processor resizes to 224",
        "n_train": int((~test).sum()), "n_test": int(test.sum()),
        "classes": names,
        "probe": {"features": "CLS vector after each block (index 0 = embeddings), standardised",
                  "classifier": "sklearn LogisticRegression, C = 0.1",
                  "accuracy_trained": acc_tr, "accuracy_random_weights": acc_rd,
                  "chance": 0.1, "best_layer": best, "best_accuracy": acc_tr[best],
                  "best_random": max(acc_rd), "best_random_layer": int(np.argmax(acc_rd))},
        "neurons": neuron_out,
        "neuron_rule": "layers 3, 7, 11; neuron = numpy default_rng(0).integers(3072), in that order; "
                       "score = max over the 49 patch tokens of the MLP hidden value (after quick-GELU)",
        "mps_vs_cpu": {"max_abs_diff_final_cls": mps_vs_cpu, "final_cls_span": cls_span, "n_images": 16},
        "device": device.type,
        "guard": guard.report(),
    })
    print(f"probe best layer {best}: {acc_tr[best]}; random best {max(acc_rd)}; MPS vs CPU {mps_vs_cpu:.2e}")
    if args.figures:
        import figures

        figures.fig_clip_probe()
        figures.fig_clip_neurons()


if __name__ == "__main__":
    main()
