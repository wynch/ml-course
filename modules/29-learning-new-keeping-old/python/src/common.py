"""Shared helpers for module 29: the laptop guard, paths, pinned data, results.

Import this module before torch. It caps the MPS allocator at half of Metal's
recommended working set (5.92 GB on a 16 GB Mac). The char-GPT and plasticity
scripts run on the CPU; only `library.py` uses the GPU, where the cap applies.
Every dataset is read from the Hugging Face cache at a pinned revision, offline.
"""

from __future__ import annotations

import os

# Must be set before torch is imported. The low mark has to sit below the
# high mark, or torch refuses to start ("invalid low watermark ratio").
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import platform
import resource
import sys
import time
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[2]          # modules/29-learning-new-keeping-old
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"
ARTIFACTS = MODULE / "artifacts"
CORPUS = MODULE / "corpus"

# Pinned as in course/offline/manifest.json.
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
SQUAD = ("rajpurkar/squad", "7b6d24c440a36b6815f21b70d25016731768db1f")
SMOLTALK = ("HuggingFaceTB/smoltalk", "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa")
FASHION = ("zalando-datasets/fashion_mnist", "531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2")

# Four threads: the same sums in the same order on any recent Mac, so a re-run
# gives the same numbers.
torch.set_num_threads(4)


def parquet(repo: tuple[str, str], filename: str) -> list[dict]:
    """Rows of one parquet file of a pinned dataset, from the local cache."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo[0], filename, repo_type="dataset", revision=repo[1])
    return pq.read_table(path).to_pylist()


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

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

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


def write_results(name: str, payload: dict, folder: Path | None = None) -> Path:
    folder = folder or RESULTS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"  wrote {path}")
    return path


def read_results(name: str, folder: Path | None = None) -> dict:
    return json.loads(((folder or RESULTS) / f"{name}.json").read_text())
