"""Shared helpers for module 22: paths, the laptop guard, results I/O.

The model runs on MLX, Apple's array library, not on torch. MLX has no
watermark setting like torch's MPS; the guard here only measures: wall time,
peak process RSS, and MLX's own peak Metal memory.
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2]          # modules/22-reasoning-at-test-time
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"
GEN = RESULTS / "generations"

MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"
# Pinned so the cached generations can be reproduced.
MODEL_REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"
DATASET_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_results(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def write_results(name: str, data: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(MODULE)}")
    return path


class Guard:
    """Wall time and peak memory for one script.

    Peak RSS comes from ``getrusage`` (bytes on macOS). Metal memory is MLX's
    own peak (``mx.get_peak_memory``), read only if MLX was imported.
    """

    def __init__(self, name: str):
        self.name = name
        self.t0 = time.perf_counter()

    def report(self) -> dict:
        secs = time.perf_counter() - self.t0
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = r if sys.platform == "darwin" else r * 1024
        metal = 0
        if "mlx.core" in sys.modules:
            metal = sys.modules["mlx.core"].get_peak_memory()
        out = {
            "script": self.name,
            "minutes": round(secs / 60, 2),
            "peak_rss_gb": round(rss / 2**30, 2),
            "peak_metal_gb": round(metal / 2**30, 2),
            "machine": f"{platform.machine()} {platform.mac_ver()[0] or platform.system()}",
        }
        print(f"[guard] {self.name}: {out['minutes']} min, peak RSS {out['peak_rss_gb']} GB, "
              f"peak Metal {out['peak_metal_gb']} GB")
        return out
