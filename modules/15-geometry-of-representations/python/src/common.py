"""Shared helpers for module 15: the laptop guard, paths, data loaders, module 02's MLP, results.

Import this module before torch. It caps the MPS allocator at half of Metal's
recommended working set (5.92 GB on a 16 GB Mac), so a run that grows too large
fails with an error instead of pushing the whole Mac into swap.
"""

from __future__ import annotations

import os

# Must be set before torch is imported. The low mark has to sit below the
# high mark, or torch refuses to start ("invalid low watermark ratio").
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import importlib.util
import io
import json
import platform
import resource
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[2]          # modules/15-geometry-of-representations
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"
M02 = MODULE.parent / "02-neural-networks" / "python" / "src"

FMNIST_ID = "zalando-datasets/fashion_mnist"
FMNIST_REVISION = "531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2"
VIT_ID = "google/vit-base-patch16-224"
VIT_REVISION = "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"
# The 160-px imagenette release from fast.ai (the archive behind frgfm/imagenette's "160px" config).
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
MLCOURSE_CACHE = Path.home() / ".cache" / "huggingface" / "mlcourse"
IMAGENETTE_DIR = MLCOURSE_CACHE / "imagenette" / "imagenette2-160"

FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free() -> None:
    """Hand MPS memory back. The caller must ``del`` its own references first."""
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class Guard:
    """Wall time and peak memory for one script.

    Peak RSS comes from ``getrusage`` (bytes on macOS, kilobytes on Linux).
    MPS memory is what the Metal driver holds for this process; we sample it
    whenever ``tick()`` is called and keep the maximum.
    """

    def __init__(self, name: str):
        self.name = name
        self.t0 = time.perf_counter()
        self.mps_peak = 0
        self.stages: dict[str, float] = {}
        self._last = self.t0

    def tick(self, stage: str | None = None) -> None:
        if torch.backends.mps.is_available():
            self.mps_peak = max(self.mps_peak, torch.mps.driver_allocated_memory())
        if stage:
            now = time.perf_counter()
            self.stages[stage] = round(now - self._last, 2)
            self._last = now

    @staticmethod
    def rss_peak_bytes() -> int:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024

    def report(self) -> dict:
        self.tick()
        secs = time.perf_counter() - self.t0
        out = {
            "script": self.name,
            "seconds": round(secs, 1),
            "minutes": round(secs / 60, 2),
            "peak_rss_gb": round(self.rss_peak_bytes() / 2**30, 2),
            "peak_mps_driver_gb": round(self.mps_peak / 2**30, 2),
            "stages_seconds": self.stages,
            "machine": f"{platform.machine()} {platform.mac_ver()[0] or platform.system()}",
            "torch": torch.__version__,
            "mps_high_watermark_ratio": os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"],
        }
        if torch.backends.mps.is_available():
            cap = torch.mps.recommended_max_memory() * float(out["mps_high_watermark_ratio"])
            out["mps_cap_gb"] = round(cap / 2**30, 2)
        print(
            f"[guard] {self.name}: {out['minutes']} min, peak RSS {out['peak_rss_gb']} GB, "
            f"peak MPS driver {out['peak_mps_driver_gb']} GB"
        )
        return out


def write_results(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(MODULE)}")
    return path


def read_results(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


# ------------------------------------------------------------------ data

def load_fashion(split: str) -> tuple[np.ndarray, np.ndarray]:
    """FashionMNIST from its Parquet file on the Hub: (N, 784) float32 in [0, 1], (N,) labels."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    path = hf_hub_download(FMNIST_ID, f"fashion_mnist/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=FMNIST_REVISION)
    t = pq.read_table(path).to_pydict()
    x = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])), dtype=np.float32) for im in t["image"]])
    return x.reshape(len(x), -1) / 255.0, np.asarray(t["label"], dtype=np.int64)


def imagenette_train_files() -> list[Path]:
    """The 9,469 training JPEGs of imagenette2-160, sorted; downloads 99 MB once if missing."""
    root = IMAGENETTE_DIR / "train"
    if not root.exists():
        dl = MLCOURSE_CACHE / "dl-imagenette"
        dl.mkdir(parents=True, exist_ok=True)
        tgz = dl / "imagenette2-160.tgz"
        if not tgz.exists():
            print(f"downloading {IMAGENETTE_URL} (99 MB) ...")
            urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(IMAGENETTE_DIR.parent, filter="data")
    return sorted(root.glob("*/*.JPEG"))


# ------------------------------------------------------------------ module 02's MLP

def _m02():
    """Module 02's NumPy package (nn, optim, train), loaded under its own name."""
    if "m02src" in sys.modules:
        return sys.modules["m02src"]
    spec = importlib.util.spec_from_file_location("m02src", M02 / "__init__.py",
                                                  submodule_search_locations=[str(M02)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["m02src"] = pkg
    spec.loader.exec_module(pkg)
    for sub in ("nn", "optim", "train"):
        importlib.import_module(f"m02src.{sub}")
    return pkg


def train_m02_mlp(x_train, y_train, seed: int, epochs: int = 25):
    """Module 02's 784 -> 256 -> 10 MLP and recipe (train_fashion.py), from its own code.

    20,000 training images drawn with the seed, 10% held out for validation, SGD lr 0.1,
    momentum 0.9, batch 128. Returns (trained model, its weights at init).
    """
    m = _m02()
    rng = np.random.default_rng(seed)
    sub = np.random.default_rng(seed).choice(len(x_train), size=20000, replace=False)
    x_tr, y_tr, x_val, y_val = m.train.train_val_split(x_train[sub], y_train[sub], val_frac=0.1, rng=rng)
    model = m.nn.MLP([784, 256, 10], rng)
    init = mlp_weights(model, copy=True)
    opt = m.optim.SGD(model.params_and_grads, lr=0.1, momentum=0.9)
    m.train.train(model, opt, x_tr, y_tr, x_val, y_val, epochs=epochs, batch_size=128, rng=rng,
                  loss_fn=m.nn.SoftmaxCrossEntropy(), verbose=False)
    return model, init


def mlp_weights(model, copy: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = [layer for layer in model.layers if hasattr(layer, "W")]
    return [(l.W.copy(), l.b.copy()) if copy else (l.W, l.b) for l in layers]


def mlp_layers(weights, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hidden ReLU outputs, logits) of a 784 -> 256 -> 10 MLP, float64."""
    (w1, b1), (w2, b2) = weights
    h = np.maximum(x.astype(np.float64) @ w1 + b1, 0)
    return h, h @ w2 + b2
