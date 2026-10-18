"""Shared helpers for module 19: the laptop guard, paths, the recipe, results I/O.

Import this module before torch. It caps the MPS allocator at half of unified
memory, so a run that grows too large fails with an error instead of pushing
the whole Mac into swap.
"""

from __future__ import annotations

import os

# Must be set before torch is imported. The low mark has to sit below the
# high mark, or torch refuses to start ("invalid low watermark ratio").
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import json
import platform
import resource
import sys
import time
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[2]          # modules/19-scaling-laws
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"

# --- the recipe, the same for every size ---------------------------------
SIZES = (32, 48, 64, 96, 160, 256)       # d_model; 4 layers, 4 heads each
TWO_SEEDS = (32, 48)                     # the two smallest also run seed 1: the noise band
N_LAYER, N_HEAD, BLOCK = 4, 4, 128
BATCH = 32                               # 32 windows x 128 characters = 4,096 tokens a step
STEPS, WARMUP = 977, 100                 # 977 steps = 4,001,792 tokens, one pass
LR, WEIGHT_DECAY = 3e-4, 0.1
N_CHARS, N_HELD = 5_000_000, 50_000      # the slice; its last 50,000 characters are held out
MIN_COUNT = 10                           # rarer characters map to one UNK id
EVAL_STEPS = (8, 15, 31, 61, 92, 122, 183, 244, 366, 488, 732, 977)
MARKED_STEPS = (61, 122, 244, 488, 977)  # 0.25, 0.5, 1, 2 and 4M tokens


def tokens_at(step: int) -> int:
    return step * BATCH * BLOCK


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def read_runs() -> dict[int, dict]:
    """Every size's training results, keyed by d_model."""
    return {d: read_results(f"train_d{d}") for d in SIZES}


def held_at(run: dict, step: int) -> float:
    return next(c["held_loss"] for c in run["curve"] if c["step"] == step)
