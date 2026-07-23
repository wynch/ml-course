"""Exercise (b) — run the attention heatmap on YOUR OWN photo.

Point this script at any image on disk. It loads google/vit-base-patch16-224,
predicts an ImageNet label, computes an attention-rollout heatmap, and saves an
overlay next to the original so you can see *where the model looked*.

    uv run python ../exercises/exercise_b_your_photo.py /path/to/your_photo.jpg

Then interpret: does the model attend to the object you care about, or to
background / texture? Try a cluttered scene and a clean studio shot and compare.

Most of the heavy lifting already lives in ``src/attention.py`` — you only need
to wire the pieces together where marked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# make src/ importable when run directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))

import attention as att


def main(image_path: str, out_path: str | None = None):
    img = Image.open(image_path).convert("RGB")

    # TODO(you): load the ViT + processor from attention.py.
    #   processor, model = ...
    # TODO(you): get attentions/logits/grid for `img`.
    #   attns, logits, grid = ...
    # TODO(you): compute the rollout heatmap and the predicted label.
    #   heat = ...
    #   pred = ...
    raise NotImplementedError

    base = img.resize((224, 224))
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    axes[0].imshow(base); axes[0].set_title("your photo"); axes[0].axis("off")
    axes[1].imshow(base)
    axes[1].imshow(att.overlay_heatmap(img, heat), cmap="jet", alpha=0.5)
    axes[1].set_title(f"attention rollout\n→ {pred}"); axes[1].axis("off")
    fig.tight_layout()
    out = out_path or str(Path(image_path).with_suffix("")) + "_attention.png"
    fig.savefig(out, bbox_inches="tight", dpi=90)
    print("wrote", out, "| prediction:", pred)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
