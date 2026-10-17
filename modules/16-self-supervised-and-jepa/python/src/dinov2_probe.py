"""Section 4: the Hugging Face way. A linear probe on frozen DINOv2-small features.

    uv run python src/dinov2_probe.py

DINOv2-small (a 22M-parameter ViT-S/14, self-supervised on 142M images) reads
CIFAR-10 images resized from 32 to 224 pixels. The probe is the same code as
section 1's, on 10,000 training images (section 1's `class_acc_10k`) and the
10,000 test images. Writes results/dinov2.json.
"""

from __future__ import annotations

import argparse

import common
import torch
import torch.nn.functional as F

import ssl_core as S

MODEL_ID = "facebook/dinov2-small"
# Pinned so the numbers in results/ can be reproduced exactly.
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
SIZE = 224
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
N_TRAIN = 10000
N_ROT = 2000          # rotation probe: train and test images, each seen at four angles


def prep(xb: torch.Tensor, rot: int = 0) -> torch.Tensor:
    x = S.to_float(xb)
    if rot:
        x = torch.rot90(x, rot, (2, 3))
    x = F.interpolate(x, size=(SIZE, SIZE), mode="bicubic", align_corners=False).clamp(0, 1)
    m = torch.tensor(IMNET_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(IMNET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def features(model, X: torch.Tensor, bs: int = 125, rot: int = 0) -> torch.Tensor:
    """CLS token after the final layer norm, joined with the mean patch token: 768 numbers per image."""
    out = []
    for i in range(0, len(X), bs):
        h = model(pixel_values=prep(X[i:i + bs], rot))
        out.append(torch.cat([h.pooler_output, h.last_hidden_state[:, 1:].mean(1)], 1).float().cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--figures", action="store_true", help="redraw every figure once all results exist")
    args = ap.parse_args()
    guard = common.Guard("dinov2")
    from transformers import AutoModel

    dev = common.get_device()
    xtr, ytr, xte, yte = common.load_cifar10()
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32).eval()
    n_params = sum(p.numel() for p in model.parameters())

    # One forward pass on the CPU and on MPS, same eight test images.
    x8 = torch.from_numpy(xte[:8])
    with torch.no_grad():
        cpu = model(pixel_values=prep(x8)).pooler_output
        model.to(dev)
        mps = model(pixel_values=prep(x8.to(dev))).pooler_output.cpu()
    mps_vs_cpu = {"images": 8, "output": "pooler_output (CLS after layer norm, 384 numbers)",
                  "max_abs_diff": float((cpu - mps).abs().max()), "output_abs_max": float(cpu.abs().max()),
                  "min_cosine": float(F.cosine_similarity(cpu, mps, dim=1).min())}
    print("MPS vs CPU", mps_vs_cpu)
    guard.tick("load")

    Xtr = torch.from_numpy(xtr[:N_TRAIN]).to(dev)
    Xte = torch.from_numpy(xte).to(dev)
    Ztr, Zte = features(model, Xtr), features(model, Xte)
    guard.tick("features")
    ytr_t, yte_t = torch.from_numpy(ytr[:N_TRAIN]), torch.from_numpy(yte)
    acc = S.logistic_probe(Ztr, ytr_t, Zte, yte_t, 10)
    col_tr, col_te = S.to_float(Xtr).mean((2, 3)).cpu(), S.to_float(Xte).mean((2, 3)).cpu()
    col = S.ridge_r2(Ztr, col_tr, Zte, col_te)
    Rtr = torch.cat([features(model, Xtr[:N_ROT], rot=k) for k in range(4)])
    Rte = torch.cat([features(model, Xte[:N_ROT], rot=k) for k in range(4)])
    r = torch.arange(4).repeat_interleave(N_ROT)
    rot = S.logistic_probe(Rtr, r, Rte, r, 4)
    guard.tick("probe")
    res = {"model": MODEL_ID, "revision": MODEL_REVISION, "params": n_params, "input_size": SIZE,
           "feature": "CLS after layer norm + mean patch token (768)", "train_images": N_TRAIN, "test_images": len(xte),
           "class_acc": round(acc, 4), "colour_r2": round(col, 4),
           "rotation_acc": round(rot, 4), "rotation_images": {"train": N_ROT, "test": N_ROT, "angles": 4},
           "mps_vs_cpu": mps_vs_cpu, "device": str(dev)}
    print({k: res[k] for k in ("class_acc", "colour_r2", "rotation_acc")})
    res["guard"] = guard.report()
    if not args.no_write:
        common.write_results("dinov2", res)
    if args.figures:
        import figures

        figures.main()


if __name__ == "__main__":
    main()
