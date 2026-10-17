"""Shared helpers for module 11: the laptop guard, paths, loaders, results.

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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import platform
import resource
import sys
import time
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[2]          # modules/11-inside-a-language-model
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"
M04 = MODULE.parent / "04-attention-transformer"

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
# Pinned so the numbers in results/ can be reproduced exactly.
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hf_model(device: str | None = None, random_weights: bool = False, seed: int = 0):
    """SmolLM2-360M-Instruct in float32, eval mode.

    ``random_weights=True`` builds the same architecture from its config with
    fresh random weights (seed fixed): the control for the probe.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    dev = get_device(device)
    if random_weights:
        cfg = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32
        )
    return model.to(dev).eval()


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def wrap(model, tokenizer):
    """Wrap a loaded Hugging Face model with nnsight 0.8."""
    from nnsight import TransformersModel

    return TransformersModel(model, tokenizer=tokenizer)


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
