# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Build the standalone ML course map.

Reads template.html, replaces every {{FIGNN}} placeholder with a data: URI of the
matching module figure (downscaled, whichever of optimized PNG / JPEG q80 is
smaller) and writes two identical copies of the finished page:

  * <repo>/map.html                     — published on GitHub Pages (tracked in git)
  * tools/coursemap/course-map.html     — scratch copy for one-off sharing (gitignored)

    uv run build_map.py
"""

from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODULES = REPO / "modules"
TEMPLATE = HERE / "template.html"
# The published page (served by GitHub Pages at /ml-course/map.html) plus a
# scratch copy next to the template. Same bytes, written in one build.
SITE_OUTPUT = REPO / "map.html"
OUTPUT = HERE / "course-map.html"
OUTPUTS = (SITE_OUTPUT, OUTPUT)

MAX_W = 680
JPEG_QUALITY = 80

# module number -> (module directory name, figure file name)
FIGS: dict[str, tuple[str, str]] = {
    "00A": ("00a-perceptron", "perceptron_evolution.png"),
    "00B": ("00b-bayes-knn-pca", "pca_eigenimages.png"),
    "00C": ("00c-kernels-hopfield", "double_descent.png"),
    "01": ("01-autograd", "computation_graph.png"),
    "02": ("02-neural-networks", "fashion_weights.png"),
    "03": ("03-tokenization", "colored_tokens.png"),
    "04": ("04-attention-transformer", "attention_heads.png"),
    "05": ("05-transformers-library", "logit_lens.png"),
    "06": ("06-fine-tuning", "before_after_panel.png"),
    "07": ("07-inference-internals", "kv_cache_timing.png"),
    "08": ("08-vision", "attention_maps.png"),
    "09": ("09-diffusion", "ddpm_dreaming.png"),
    "10": ("10-agents", "handrolled_trace.png"),
}


def data_uri(path: Path) -> tuple[str, int]:
    """Downscale `path` and return (data URI, encoded byte length)."""
    img = Image.open(path)
    img.load()
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA")
    if img.width > MAX_W:
        height = round(img.height * MAX_W / img.width)
        img = img.resize((MAX_W, height), Image.LANCZOS)

    png_buf = io.BytesIO()
    img.save(png_buf, format="PNG", optimize=True)

    flat = img
    if flat.mode in ("RGBA", "LA", "P"):
        flat = flat.convert("RGBA")
        bg = Image.new("RGB", flat.size, (255, 255, 255))
        bg.paste(flat, mask=flat.split()[-1])
        flat = bg
    elif flat.mode != "RGB":
        flat = flat.convert("RGB")
    jpg_buf = io.BytesIO()
    flat.save(jpg_buf, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)

    png, jpg = png_buf.getvalue(), jpg_buf.getvalue()
    raw, mime = (png, "image/png") if len(png) <= len(jpg) else (jpg, "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", len(raw)


def main() -> int:
    html = TEMPLATE.read_text(encoding="utf-8")
    total = 0

    for num, (module_dir, figure) in FIGS.items():
        src = MODULES / module_dir / "figures" / figure
        if not src.exists():
            print(f"missing figure: {src}", file=sys.stderr)
            return 1
        uri, size = data_uri(src)
        token = "{{FIG%s}}" % num
        if token not in html:
            print(f"placeholder {token} not found in template", file=sys.stderr)
            return 1
        html = html.replace(token, uri)
        total += size
        print(f"  {num}  {module_dir}/figures/{figure:<24} {size / 1024:7.1f} KB")

    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", html)
    if left:
        print(f"unresolved placeholders: {sorted(set(left))}", file=sys.stderr)
        return 1

    for out in OUTPUTS:
        out.write_text(html, encoding="utf-8")

    size = SITE_OUTPUT.stat().st_size / 1024
    print(
        f"\n{size:.0f} KB ({len(FIGS)} figures, {total / 1024:.0f} KB of image data)"
    )
    for out in OUTPUTS:
        print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
