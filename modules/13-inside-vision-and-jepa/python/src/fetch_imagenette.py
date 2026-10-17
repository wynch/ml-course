"""Download imagenette2-160 (99 MB) once and unpack it under ~/.cache/huggingface/mlcourse/.

imagenette is ten easy ImageNet classes, published by fast.ai (Apache 2.0).
The 160 px version has the shorter side of every image at 160 pixels. The
Hugging Face copy (frgfm/imagenette) is a loading script that fetches this same
file, so we fetch it directly and check its SHA-256.

    uv run python src/fetch_imagenette.py
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request

from common import IMAGENETTE, IMAGENETTE_SHA256, IMAGENETTE_URL


def main() -> None:
    if (IMAGENETTE / "val").is_dir():
        print(f"already there: {IMAGENETTE}")
        return
    root = IMAGENETTE.parent
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / "imagenette2-160.tgz"
    if not tgz.exists():
        print(f"downloading {IMAGENETTE_URL} (99 MB)")
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
    digest = hashlib.sha256(tgz.read_bytes()).hexdigest()
    if digest != IMAGENETTE_SHA256:
        raise SystemExit(f"SHA-256 mismatch for {tgz}: {digest}")
    with tarfile.open(tgz) as tf:
        tf.extractall(root, filter="data")      # refuses absolute paths and links out of root
    tgz.unlink()
    print(f"unpacked to {IMAGENETTE}")


if __name__ == "__main__":
    main()
