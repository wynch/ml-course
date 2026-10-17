"""Section 2: intrinsic dimension, on sets of known dimension and then on real data.

1. TwoNN and MLE on curved cubes of dimension 2, 5, 10 and 20 in 784-d (we know the answer).
2. FashionMNIST's 10,000 test images: pixels, then module 02's MLP (hidden layer and logits),
   trained and at init.
3. ViT-base's CLS vector on 5,000 imagenette images, at every layer. One MPS-vs-CPU check.

    uv run python src/intrinsic_dim.py --figures
"""

from __future__ import annotations

import argparse

import common
import idest
import numpy as np
import torch

KNOWN_DIMS = (2, 5, 10, 20)
N = 10_000
N_VIT = 5_000


def run_known() -> list[dict]:
    rows = []
    for d in KNOWN_DIMS:
        e = idest.estimate(idest.curved_cube(d, N, seed=d))
        rows.append({"true_dim": d, **e})
        print(f"curved cube d={d}: TwoNN {e['twonn']}, MLE {e['mle']}")
    return rows


def run_fashion() -> dict:
    x_tr, y_tr = common.load_fashion("train")
    x_te, y_te = common.load_fashion("test")
    model, init = common.train_m02_mlp(x_tr, y_tr, seed=0)
    acc = common._m02().nn.accuracy(model, x_te, y_te)
    h, z = common.mlp_layers(common.mlp_weights(model), x_te)
    h0, z0 = common.mlp_layers(init, x_te)
    out = {"mlp_test_acc": round(float(acc), 4)}
    for name, x in (("pixels", x_te.astype(np.float64)), ("hidden", h), ("logits", z),
                    ("hidden_init", h0), ("logits_init", z0)):
        out[name] = idest.estimate(x)
        print(f"FashionMNIST {name}: TwoNN {out[name]['twonn']}, MLE {out[name]['mle']}")
    return out


def run_vit(guard) -> dict:
    from PIL import Image
    from transformers import ViTImageProcessor, ViTModel

    files = common.imagenette_train_files()
    idx = np.sort(np.random.default_rng(0).choice(len(files), size=N_VIT, replace=False))
    images = [Image.open(files[i]).convert("RGB") for i in idx]
    guard.tick("vit images")
    proc = ViTImageProcessor.from_pretrained(common.VIT_ID, revision=common.VIT_REVISION)
    model = ViTModel.from_pretrained(common.VIT_ID, revision=common.VIT_REVISION, add_pooling_layer=False).eval()
    dev = common.get_device()
    with torch.no_grad():
        px = proc(images=images[:8], return_tensors="pt")["pixel_values"]
        cpu_cls = model(pixel_values=px).last_hidden_state[:, 0]
        model.to(dev)
        dev_cls = model(pixel_values=px.to(dev)).last_hidden_state[:, 0].cpu()
        check = {"images": 8, "max_abs_diff_cls": float((cpu_cls - dev_cls).abs().max()),
                 "cls_abs_max": float(cpu_cls.abs().max()), "device": dev.type}
        per_layer, last = [], []
        for s in range(0, len(images), 50):
            px = proc(images=images[s:s + 50], return_tensors="pt")["pixel_values"].to(dev)
            o = model(pixel_values=px, output_hidden_states=True)
            per_layer.append(torch.stack([hs[:, 0] for hs in o.hidden_states], 1).cpu().double().numpy())
            last.append(o.last_hidden_state[:, 0].cpu().double().numpy())
            guard.tick()
    del model
    common.free()
    guard.tick("vit forward")
    per_layer, last = np.concatenate(per_layer), np.concatenate(last)
    layers = []
    # hidden_states[0] holds the CLS token before any block: the same vector for every image.
    assert np.unique(per_layer[:, 0], axis=0).shape[0] == 1
    for layer in range(1, per_layer.shape[1]):
        e = idest.estimate(per_layer[:, layer])
        layers.append({"layer": layer, "twonn": e["twonn"], "mle": e["mle"]})
    cls = idest.estimate(last)
    print(f"ViT CLS (after final norm): TwoNN {cls['twonn']}, MLE {cls['mle']}; MPS vs CPU {check['max_abs_diff_cls']:.2e}")
    pixels = idest.estimate(np.stack([np.asarray(im.resize((64, 64)), dtype=np.float64).ravel() / 255
                                      for im in images]))
    print(f"imagenette pixels at 64x64: TwoNN {pixels['twonn']}, MLE {pixels['mle']}")
    return {"n_images": N_VIT, "source": "imagenette2-160, train split, numpy seed 0", "cls": cls,
            "per_layer_cls": layers, "pixels_64": {k: pixels[k] for k in ("n", "twonn", "mle")},
            "mps_vs_cpu": check}


def fig_id(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    k = r["known"]
    d = [row["true_dim"] for row in k]
    ax.plot(d, d, color="0.7", lw=1, label="true dimension")
    ax.plot(d, [row["twonn"] for row in k], "o-", color="#1d6fb8", label="TwoNN")
    ax.plot(d, [row["mle"] for row in k], "s-", color="#b7410e", label="MLE, k 10–20")
    ax.set_xlabel("true dimension of the curved cube")
    ax.set_ylabel("estimate")
    ax.set_title(f"Known sets in 784-d, n = {k[0]['n']:,}", fontsize=10)
    ax.legend(fontsize=9)
    ax = axes[1]
    f = r["fashion"]
    names = ["pixels", "hidden", "logits"]
    xs = np.arange(3)
    ax.bar(xs - 0.3, [f[n]["twonn"] for n in names], 0.3, color="#1d6fb8", label="TwoNN, trained")
    ax.bar(xs, [f[n]["mle"] for n in names], 0.3, color="#b7410e", label="MLE, trained")
    ax.bar(xs[1:] + 0.3, [f[n + "_init"]["mle"] for n in names[1:]], 0.3, color="0.75", label="MLE, at init")
    ax.set_xticks(xs, ["pixels (784)", "hidden (256)", "logits (10)"])
    ax.set_title("FashionMNIST test set through module 02's MLP", fontsize=10)
    ax.legend(fontsize=9)
    ax = axes[2]
    v = r["vit"]["per_layer_cls"]
    ax.plot([row["layer"] for row in v], [row["twonn"] for row in v], "o-", color="#1d6fb8", label="TwoNN")
    ax.plot([row["layer"] for row in v], [row["mle"] for row in v], "s-", color="#b7410e", label="MLE")
    ax.set_xlabel("ViT-base block (CLS after block l)")
    ax.set_title(f"CLS vector, {r['vit']['n_images']:,} imagenette images", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = common.Guard("intrinsic_dim.py")
    torch.set_num_threads(4)
    known = run_known()
    guard.tick("known sets")
    fashion = run_fashion()
    guard.tick("fashion")
    vit = run_vit(guard)
    res = {"known": known, "fashion": fashion, "vit": vit, "guard": guard.report()}
    common.write_results("intrinsic_dim", res)
    if args.figures:
        fig_id(res, common.FIGURES / "intrinsic_dim.png")


if __name__ == "__main__":
    main()
