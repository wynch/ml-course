"""Reference solution for exercise (b) — attention heatmap on your own photo."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# make src/ importable when run directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))

import attention as att


def run(image_path: str, out_path: str | None = None) -> tuple[str, str]:
    img = Image.open(image_path).convert("RGB")

    processor, model = att.load_vit()
    attns, logits, grid = att.get_attentions(processor, model, img)
    heat = att.attention_rollout(attns, grid)
    pred = att.top_label(model, logits)

    base = img.resize((224, 224))
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    axes[0].imshow(base); axes[0].set_title("your photo"); axes[0].axis("off")
    axes[1].imshow(base)
    axes[1].imshow(att.overlay_heatmap(img, heat), cmap="jet", alpha=0.5)
    axes[1].set_title(f"attention rollout\n→ {pred}"); axes[1].axis("off")
    fig.tight_layout()
    out = out_path or str(Path(image_path).with_suffix("")) + "_attention.png"
    fig.savefig(out, bbox_inches="tight", dpi=90)
    plt.close(fig)
    print("wrote", out, "| prediction:", pred)
    return out, pred


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Fall back to the bundled Grace Hopper photo so the solution is runnable
        # with no arguments (written to a temp dir, never into the repo).
        import tempfile

        from utils import grace_hopper

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "grace_hopper.jpg"
            grace_hopper().save(p)
            run(str(p), str(Path(tmp) / "out.png"))
    else:
        run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
