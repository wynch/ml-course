"""Shared helpers for module 33: the laptop guard, paths, results.

Import this module before torch. It caps the MPS allocator at 0.3 of Metal's
recommended working set (3.55 GB on a 16 GB Mac), under the module's 4 GB
budget. Training and sampling run on the GPU (MPS), where the cap applies; the
2-D flow-matching section and the tests use the CPU.
"""

from __future__ import annotations

import os

# Must be set before torch is imported. The low mark has to sit below the
# high mark, or torch refuses to start ("invalid low watermark ratio").
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")

import json
import platform
import resource
import sys
import time
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[2]          # modules/33-text-by-denoising
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"
ARTIFACTS = MODULE / "artifacts"
RUNS = ARTIFACTS / "runs"                             # full checkpoints, git-ignored
CORPUS = MODULE / "corpus" / "tiny_shakespeare.txt"
CHUNK_SECONDS = 270                                   # no process trains longer than this

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


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def log_guard(rep: dict, folder: str = "guards") -> None:
    """Append one process's guard report to results/<folder>.jsonl."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / f"{folder}.jsonl").open("a") as fh:
        fh.write(json.dumps(rep) + "\n")


def write_results(name: str, payload: dict, folder: Path | None = None) -> Path:
    folder = folder or RESULTS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"  wrote {path}")
    return path


def read_results(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())
