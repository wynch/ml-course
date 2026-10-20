"""Shared helpers for module 16: the laptop guard, paths, CIFAR-10, results.

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

import io
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[2]          # modules/16-self-supervised-and-jepa
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"

CIFAR_ID = "uoft-cs/cifar10"
# Pinned so the numbers in results/ can be reproduced exactly.
CIFAR_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CACHE = Path.home() / ".cache" / "huggingface" / "mlcourse"


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(x_train, y_train, x_test, y_test); images uint8 (N, 32, 32, 3).

    The first call decodes the 60,000 PNGs of the cached parquet files (a few
    seconds) and keeps a 185 MB .npz beside the Hugging Face cache.
    """
    npz = CACHE / f"cifar10_{CIFAR_REVISION[:8]}.npz"
    if npz.exists():
        d = np.load(npz)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    def split(name):
        f = hf_hub_download(CIFAR_ID, f"plain_text/{name}-00000-of-00001.parquet", repo_type="dataset",
                            revision=CIFAR_REVISION)
        t = pq.read_table(f)
        imgs = t.column("img").combine_chunks().field("bytes").to_pylist()
        x = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in imgs])
        return x, np.asarray(t.column("label").to_pylist(), dtype=np.int64)

    xtr, ytr = split("train")
    xte, yte = split("test")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(npz, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte


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
            # The cap the ratio sets: a share of Metal's recommended working set.
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
