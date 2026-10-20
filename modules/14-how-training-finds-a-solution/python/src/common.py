"""Shared helpers for module 14: the laptop guard, paths, loaders, results.

Import this module before torch. It caps the MPS allocator at half of Metal's
recommended working set (5.92 GB on a 16 GB Mac), so a run that grows too
large fails with an error instead of pushing the whole Mac into swap.
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

import torch

MODULE = Path(__file__).resolve().parents[2]          # modules/14-how-training-finds-a-solution
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"

FASHION_ID = "zalando-datasets/fashion_mnist"


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_fashion(train_subsample: int = 20000, seed: int = 0):
    """FashionMNIST as float tensors in [0, 1], flattened to 784.

    Same choice as module 02: a 20,000-image training subsample drawn with
    ``numpy.random.default_rng(seed)``, and the full 10,000-image test set.
    """
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(FASHION_ID)

    def arrays(split):
        x = np.stack([np.asarray(im, dtype=np.float32) for im in split["image"]]).reshape(len(split), -1) / 255.0
        return torch.from_numpy(x), torch.tensor(split["label"], dtype=torch.long)

    x_tr, y_tr = arrays(ds["train"])
    x_te, y_te = arrays(ds["test"])
    idx = np.random.default_rng(seed).choice(len(x_tr), size=train_subsample, replace=False)
    return x_tr[idx], y_tr[idx], x_te, y_te


class MLP(torch.nn.Module):
    """Module 02's network in torch: 784 -> 256 -> ReLU -> 10, He-initialised weights, zero biases."""

    def __init__(self, sizes=(784, 256, 10)):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            lin = torch.nn.Linear(sizes[i], sizes[i + 1])
            torch.nn.init.normal_(lin.weight, 0.0, (2.0 / sizes[i]) ** 0.5)
            torch.nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < len(sizes) - 2:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, x, y, batch: int = 5000) -> tuple[float, float]:
    """Mean cross-entropy and accuracy over (x, y)."""
    loss, correct = 0.0, 0
    for i in range(0, len(x), batch):
        out = model(x[i:i + batch])
        loss += torch.nn.functional.cross_entropy(out, y[i:i + batch], reduction="sum").item()
        correct += (out.argmax(-1) == y[i:i + batch]).sum().item()
    return loss / len(x), correct / len(x)


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
