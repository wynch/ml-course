"""Generate every figure embedded in the module README.

Run all of it with ``uv run python src/figures.py`` (retrains the tiny CNN and
the beans ViT, so budget a few minutes). Individual figures can be produced by
name: ``uv run python src/figures.py conv learned patches attention``.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import conv
import utils
from utils import savefig, use_style


# ===========================================================================
# 1a. Classic hand-designed kernels
# ===========================================================================
def fig_conv_kernels():
    img = utils.to_gray_array(utils.grace_hopper(size=256))
    kernels = conv.KERNELS
    names = list(kernels.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 3, figsize=(6.6, 2.0 * n))
    for r, name in enumerate(names):
        k = kernels[name]
        out = conv.conv2d(img, k, padding="same")
        # input
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center", labelpad=45)
        # kernel
        axes[r, 1].imshow(k, cmap="RdBu_r", vmin=-abs(k).max(), vmax=abs(k).max())
        for (i, j), v in np.ndenumerate(k):
            axes[r, 1].text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=6.5)
        # output
        axes[r, 2].imshow(conv.normalize01(out), cmap="gray")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_title("input")
    axes[0, 1].set_title("kernel")
    axes[0, 2].set_title("output")
    fig.suptitle("Hand-designed kernels: convolution as a sliding weighted sum", y=1.0)
    fig.tight_layout()
    p = savefig(fig, "conv_kernels.png", dpi=85)
    plt.close(fig)
    return p


# ===========================================================================
# 1b. Learned kernels — a tiny CNN invents its own edge detectors
# ===========================================================================
def fig_learned_filters(model=None):
    import cnn

    if model is None:
        model, _, acc = cnn.train_tiny_cnn()
    model = model.to("cpu").eval()

    # first-layer filters (8 x 1 x 5 x 5)
    filters = model.conv1.weight.detach().numpy()[:, 0]  # (8, 5, 5)

    # feature maps for one input
    (_, _), (xte, _) = cnn.load_fashion_subset(n_train=64, n_test=8)
    x = xte[0:1]  # (1,1,28,28)
    with torch.no_grad():
        a1, _ = model.features(x)
    a1 = a1[0].numpy()  # (8, 28, 28)

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.4, 1, 1], hspace=0.35, wspace=0.15)

    # input image spanning left
    ax_in = fig.add_subplot(gs[0, :2])
    ax_in.imshow(x[0, 0], cmap="gray"); ax_in.set_title("input", fontsize=10)
    ax_in.set_xticks([]); ax_in.set_yticks([])

    ax_txt = fig.add_subplot(gs[0, 2:])
    ax_txt.axis("off")
    ax_txt.text(0.0, 0.5,
                "Nobody hand-picked these kernels.\nGradient descent discovered them —\n"
                "oriented edge & blob detectors emerge\non their own. Below: the 8 learned\n"
                "5×5 filters and the feature map each\nproduces for the input at left.",
                fontsize=9.5, va="center")

    vmax = np.abs(filters).max()
    for j in range(8):
        axf = fig.add_subplot(gs[1, j])
        axf.imshow(filters[j], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axf.set_xticks([]); axf.set_yticks([])
        if j == 0:
            axf.set_ylabel("filters", fontsize=9)
        axm = fig.add_subplot(gs[2, j])
        axm.imshow(a1[j], cmap="magma")
        axm.set_xticks([]); axm.set_yticks([])
        if j == 0:
            axm.set_ylabel("feature\nmaps", fontsize=9)
    fig.suptitle("Learned kernels: the network invents its own edge detectors", y=0.98)
    p = savefig(fig, "learned_filters.png")
    plt.close(fig)
    return p, model


# ===========================================================================
# 2. ViT patchification: a photo becomes a sequence of patch tokens
# ===========================================================================
def fig_patches():
    from vit_anatomy import patchify

    P = 16
    img = utils.grace_hopper(size=224)
    arr = np.asarray(img)
    grid = patchify(arr, P)          # (14,14,16,16,3)
    n_h, n_w = grid.shape[:2]

    fig = plt.figure(figsize=(7.6, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.25)

    # (a) image with grid lines
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(arr)
    for i in range(1, n_h):
        ax0.axhline(i * P - 0.5, color="w", lw=0.5)
        ax0.axvline(i * P - 0.5, color="w", lw=0.5)
    ax0.set_title(f"224×224 image\n+ {P}×{P} grid", fontsize=10)
    ax0.set_xticks([]); ax0.set_yticks([])

    # (b) exploded patches (spaced grid)
    ax1 = fig.add_subplot(gs[1])
    gap = 3
    canvas = np.ones((n_h * (P + gap), n_w * (P + gap), 3), dtype=np.uint8) * 255
    for i in range(n_h):
        for j in range(n_w):
            y0, x0 = i * (P + gap), j * (P + gap)
            canvas[y0:y0 + P, x0:x0 + P] = grid[i, j]
    ax1.imshow(canvas)
    ax1.set_title(f"{n_h}×{n_w} = {n_h*n_w} patches\n(exploded)", fontsize=10)
    ax1.set_xticks([]); ax1.set_yticks([])

    # (c) the sequence: first patches as tokens in a row
    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    seq = grid.reshape(n_h * n_w, P, P, 3)
    show = 6
    for t in range(show):
        ax = ax2.inset_axes([0.0, 1 - (t + 1) * 0.14, 0.28, 0.12])
        ax.imshow(seq[t]); ax.set_xticks([]); ax.set_yticks([])
        ax2.text(0.32, 1 - (t + 1) * 0.14 + 0.06,
                 f"→ linear proj → patch token {t+1}", fontsize=8.5, va="center")
    ax2.text(0.0, 1 - (show + 1) * 0.14 + 0.05, "⋮   (196 tokens + 1 [CLS])", fontsize=9)
    ax2.set_title("… as a token sequence\n(just like BPE tokens, module 03)", fontsize=10)
    fig.suptitle("A Vision Transformer 'tokenizes' an image into patches", y=1.02)
    p = savefig(fig, "patch_grid.png")
    plt.close(fig)
    return p


# ===========================================================================
# 3. Attention on images — where does the model look?
# ===========================================================================
def fig_attention():
    import attention as att

    processor, model = att.load_vit()
    leaf, leaf_lbl = utils.beans_samples(1, seed=3)[0]
    photos = [
        ("Grace Hopper", utils.grace_hopper()),
        ("cat", utils.cats_image()),
        (f"bean leaf ({leaf_lbl})", leaf),
    ]

    n = len(photos)
    fig, axes = plt.subplots(n, 3, figsize=(6.6, 2.2 * n))
    for r, (name, img) in enumerate(photos):
        attns, logits, grid = att.get_attentions(processor, model, img)
        pred = att.top_label(model, logits)
        cls = att.cls_attention_map(attns, grid)
        roll = att.attention_rollout(attns, grid)
        base = img.convert("RGB").resize((224, 224))

        axes[r, 0].imshow(base)
        axes[r, 0].set_ylabel(f"{name}\n→ {pred}", fontsize=8.5, rotation=0,
                              ha="right", va="center", labelpad=52)
        axes[r, 1].imshow(base)
        axes[r, 1].imshow(att.overlay_heatmap(img, cls), cmap="jet", alpha=0.5)
        axes[r, 2].imshow(base)
        axes[r, 2].imshow(att.overlay_heatmap(img, roll), cmap="jet", alpha=0.5)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_title("photo")
    axes[0, 1].set_title("last-layer [CLS] attention")
    axes[0, 2].set_title("attention rollout")
    fig.suptitle("Where does the ViT look? (google/vit-base-patch16-224)", y=1.0)
    fig.tight_layout()
    p = savefig(fig, "attention_maps.png", dpi=90)
    plt.close(fig)
    return p


# ===========================================================================
# 4. Fine-tune figures: samples, curves, confusion, predictions
# ===========================================================================
def fig_beans_samples():
    samples = utils.beans_samples(9, seed=1)
    fig, axes = plt.subplots(3, 3, figsize=(5.4, 5.6))
    for ax, (img, lbl) in zip(axes.flat, samples):
        ax.imshow(img.resize((150, 150)))
        ax.set_title(lbl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("beans dataset — 3 classes of leaf health", y=0.99)
    fig.tight_layout()
    p = savefig(fig, "beans_samples.png", dpi=72)
    plt.close(fig)
    return p


def fig_training_and_eval(res):
    # --- training curves ---
    train_loss = [(h["epoch"], h["loss"]) for h in res.history if "loss" in h]
    eval_acc = [(h["epoch"], h["eval_accuracy"]) for h in res.history if "eval_accuracy" in h]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.4))
    ep, loss = zip(*train_loss)
    ax1.plot(ep, loss, "o-", color="C3", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="C3")
    ax1.tick_params(axis="y", labelcolor="C3")
    ax2 = ax1.twinx()
    ep2, acc = zip(*eval_acc)
    ax2.plot(ep2, acc, "s-", color="C0", label="val accuracy")
    ax2.set_ylabel("val accuracy", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    ax2.set_ylim(0, 1.02)
    ax1.set_title(f"Fine-tuning ViT-tiny on beans (test acc {res.test_acc:.1%})")
    fig.tight_layout()
    p_curves = savefig(fig, "training_curves.png")
    plt.close(fig)

    # --- confusion matrix ---
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(res.y_true, res.y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(res.class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(res.class_names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    thr = cm.max() / 2
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > thr else "black", fontsize=11)
    ax.set_title("Confusion matrix (beans test set)")
    fig.tight_layout()
    p_cm = savefig(fig, "confusion_matrix.png")
    plt.close(fig)
    return p_curves, p_cm


def fig_predictions(res):
    """One correct example per class + one honest mistake, with confidences."""
    from datasets import load_dataset

    ds = load_dataset(utils.BEANS, split="test")
    names = res.class_names
    picks = []
    for c in range(3):
        # a confident correct prediction for class c
        mask = (res.y_true == c) & (res.y_pred == c)
        idxs = np.where(mask)[0]
        if len(idxs):
            best = idxs[np.argmax(res.y_conf[idxs])]
            picks.append(("correct", best))
    # one mistake, if any
    wrong = np.where(res.y_true != res.y_pred)[0]
    if len(wrong):
        picks.append(("mistake", wrong[np.argmax(res.y_conf[wrong])]))

    fig, axes = plt.subplots(1, len(picks), figsize=(2.4 * len(picks), 3.0))
    if len(picks) == 1:
        axes = [axes]
    for ax, (kind, i) in zip(axes, picks):
        ax.imshow(ds[int(i)]["image"].resize((140, 140)))
        t, pr, cf = names[res.y_true[i]], names[res.y_pred[i]], res.y_conf[i]
        color = "green" if kind == "correct" else "red"
        ax.set_title(f"true: {t}\npred: {pr} ({cf:.0%})", fontsize=8.5, color=color)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Per-class predictions with confidence", y=1.02)
    fig.tight_layout()
    p = savefig(fig, "predictions.png", dpi=78)
    plt.close(fig)
    return p


# ===========================================================================
# 5. CLIP zero-shot similarity matrix
# ===========================================================================
def fig_clip(res_ft_acc=None):
    from clip_zeroshot import SHORT, zero_shot_beans

    out = zero_shot_beans(n=12, seed=2)   # small readable matrix
    sim = out["sim"]
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    im = ax.imshow(sim, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(SHORT))); ax.set_xticklabels(SHORT, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(out["y_true"])))
    ax.set_yticklabels([f"img{i} ({out['class_names'][t]})" for i, t in enumerate(out["y_true"])], fontsize=7)
    ax.set_xlabel("text prompt")
    for i in range(sim.shape[0]):
        j = int(sim[i].argmax())
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine similarity")
    full = zero_shot_beans(seed=0)["accuracy"]
    title = f"CLIP zero-shot similarity (red = model's pick)\nfull test acc {full:.1%}"
    if res_ft_acc is not None:
        title += f"  vs fine-tuned {res_ft_acc:.1%}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    p = savefig(fig, "clip_similarity.png")
    plt.close(fig)
    return p, full


# ===========================================================================
# Orchestration
# ===========================================================================
def main(which):
    use_style()
    done = []
    ft_res = None
    ft_acc = None

    if "conv" in which:
        done.append(fig_conv_kernels())
    if "learned" in which:
        p, _ = fig_learned_filters()
        done.append(p)
    if "patches" in which:
        done.append(fig_patches())
    if "attention" in which:
        done.append(fig_attention())
    if "samples" in which:
        done.append(fig_beans_samples())

    needs_ft = {"curves", "confusion", "predictions"} & set(which)
    if needs_ft:
        import finetune
        ft_res = finetune.train_beans()
        ft_acc = ft_res.test_acc
        if "curves" in which or "confusion" in which:
            c, m = fig_training_and_eval(ft_res)
            done += [c, m]
        if "predictions" in which:
            done.append(fig_predictions(ft_res))

    if "clip" in which:
        p, clip_acc = fig_clip(ft_acc)
        done.append(p)
        print(f"[clip] zero-shot full-test accuracy: {clip_acc:.4f}")

    if ft_acc is not None:
        print(f"[finetune] beans test accuracy: {ft_acc:.4f}")
    for p in done:
        print("wrote", p)


ALL = ["conv", "learned", "patches", "attention", "samples",
       "curves", "confusion", "predictions", "clip"]

if __name__ == "__main__":
    which = sys.argv[1:] or ALL
    if which == ["all"]:
        which = ALL
    main(which)
