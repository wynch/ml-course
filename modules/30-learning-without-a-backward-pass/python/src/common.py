"""Shared helpers for module 30: the laptop guard, paths, data, results.

Import this module before torch. It caps the MPS allocator at half of Metal's
recommended working set (5.92 GB on a 16 GB Mac), so a run that grows too
large fails with an error instead of pushing the whole Mac into swap. The
FashionMNIST sections run on the CPU; only the library lane uses MPS.
"""

from __future__ import annotations

import os

# Must be set before torch is imported. The low mark has to sit below the
# high mark, or torch refuses to start ("invalid low watermark ratio").
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[2]          # modules/30-learning-without-a-backward-pass
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"
RUNS = RESULTS / "runs"

FASHION_ID = "zalando-datasets/fashion_mnist"
FASHION_REVISION = "531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2"     # the course's pin (offline/manifest.json)
CACHE = MODULE / "python" / ".runs"                                  # git-ignored: the decoded arrays

# Four threads: the same sums in the same order on any recent Mac, so a re-run
# gives the same numbers.
torch.set_num_threads(4)


class Guard:
    """Wall time and peak memory for one script.

    Peak RSS comes from ``getrusage`` (bytes on macOS, kilobytes on Linux).
    MPS memory is what the Metal driver holds for this process; it stays 0
    for the CPU scripts.
    """

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device
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
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "device": self.device,
            "mps_high_watermark_ratio": os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"],
        }
        print(
            f"[guard] {self.name}: {out['minutes']} min, peak RSS {out['peak_rss_gb']} GB, "
            f"peak MPS driver {out['peak_mps_driver_gb']} GB"
        )
        return out


def load_fashion():
    """FashionMNIST as float32 tensors in [0, 1], flattened to 784: all 60,000
    training and 10,000 test images. Decoded once into ``python/.runs/``."""
    path = CACHE / "fashion.npz"
    if not path.exists():
        from datasets import load_dataset

        ds = load_dataset(FASHION_ID, revision=FASHION_REVISION)

        def arrays(split):
            x = np.stack([np.asarray(im, dtype=np.uint8) for im in split["image"]]).reshape(len(split), -1)
            return x, np.asarray(split["label"], dtype=np.int64)

        x_tr, y_tr = arrays(ds["train"])
        x_te, y_te = arrays(ds["test"])
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te)
    d = np.load(path)
    f = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0)
    return f(d["x_tr"]), torch.from_numpy(d["y_tr"]), f(d["x_te"]), torch.from_numpy(d["y_te"])


def write_results(name: str, payload: dict, folder: Path | None = None) -> Path:
    folder = folder or RESULTS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"  wrote {path}")
    return path


def read_results(name: str, folder: Path | None = None) -> dict:
    return json.loads(((folder or RESULTS) / f"{name}.json").read_text())
