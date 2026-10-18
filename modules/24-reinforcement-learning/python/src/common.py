"""Shared helpers for module 24: the laptop guard, paths, results.

Import this module before torch. It caps the MPS allocator at half of unified
memory, so a run that grows too large fails with an error instead of pushing
the whole Mac into swap. Every network in this module runs on the CPU, so the
cap is only a guard.
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

MODULE = Path(__file__).resolve().parents[2]          # modules/24-reinforcement-learning
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"


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


def _dumps(obj, depth: int = 0) -> str:
    """JSON with one line per dict key; a list holding no dict goes on one line."""
    pad = " " * (depth + 1)
    if isinstance(obj, dict) and obj:
        items = [f"{pad}{json.dumps(k, ensure_ascii=False)}: {_dumps(v, depth + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"
    if isinstance(obj, list) and any(isinstance(x, dict) for x in obj):
        return "[\n" + ",\n".join(pad + _dumps(x, depth + 1) for x in obj) + "\n" + " " * depth + "]"
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_results(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(_dumps(payload) + "\n")
    print(f"  wrote {path.relative_to(MODULE)}")
    return path


def read_results(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def curve_on_grid(ends, returns, grid, window: int = 20):
    """Mean return of the last ``window`` episodes finished by each step in ``grid`` (None before the first)."""
    import numpy as np

    ends, returns = np.asarray(ends), np.asarray(returns)
    out = []
    for g in grid:
        k = int(np.searchsorted(ends, g, side="right"))
        out.append(round(float(returns[max(0, k - window):k].mean()), 1) if k else None)
    return out
