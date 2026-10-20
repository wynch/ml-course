"""Shared helpers for module 13: the laptop guard, paths, image loaders, results.

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

import json
import platform
import resource
import sys
import time
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[2]          # modules/13-inside-vision-and-jepa
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"
ARTIFACTS = MODULE / "artifacts"                       # gridjepa encoder and frames, committed

HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
# imagenette2-160 from fast.ai, unpacked by fetch_imagenette.py.
IMAGENETTE = Path.home() / ".cache" / "huggingface" / "mlcourse" / "imagenette" / "imagenette2-160"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
IMAGENETTE_CLASSES = {
    "n01440764": "tench", "n02102040": "springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}

CLIP_ID = "openai/clip-vit-base-patch32"
# Pinned so the numbers in results/ can be reproduced exactly.
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


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


def need_imagenette() -> None:
    if not (IMAGENETTE / "val").is_dir():
        sys.exit(f"imagenette is missing at {IMAGENETTE}; run: uv run python src/fetch_imagenette.py")


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


def write_results(name: str, payload: dict, compact: bool = False) -> Path:
    """results/<name>.json; ``compact`` drops the indent (for long arrays)."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) if compact else \
        json.dumps(payload, indent=1, ensure_ascii=False)
    path.write_text(text + "\n")
    print(f"  wrote {path.relative_to(MODULE)}")
    return path


def read_results(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())
