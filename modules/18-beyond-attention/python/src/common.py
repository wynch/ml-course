"""Shared helpers for module 18: the laptop guard, paths, char data, results.

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

MODULE = Path(__file__).resolve().parents[2]          # modules/18-beyond-attention
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"
M04 = MODULE.parent / "04-attention-transformer"



def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device) -> None:
    """Wait for queued GPU work, so wall-clock timings mean something."""
    if torch.device(device).type == "mps":
        torch.mps.synchronize()


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


# ------------------------------------------------------------ char data
# Module 04's corpus and split: the last 10% of tiny-shakespeare is validation.
M04_ARTIFACTS = M04 / "artifacts"
CORPUS = M04 / "corpus" / "tiny_shakespeare.txt"
BLOCK = 128          # context length of every training run, as in module 04
BATCH = 64           # batch size of every training run, as in module 04


class CharData:
    """Module 04's char vocabulary (65 symbols) and its train/val split."""

    def __init__(self):
        self.chars = json.loads((M04_ARTIFACTS / "tokenizer_chars.json").read_text())["chars"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        text = CORPUS.read_text()
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n_val = int(len(data) * 0.1)
        self.train, self.val = data[:-n_val], data[-n_val:]
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def batch(self, split: str, gen: torch.Generator, device, batch: int = BATCH, block: int = BLOCK):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block] for i in ix])
        return x.to(device), y.to(device)

    def val_windows(self, n: int = 100, batch: int = BATCH, block: int = BLOCK):
        """A fixed set of n validation batches (seed 0), the same for every model."""
        gen = torch.Generator().manual_seed(0)
        return [self.batch("val", gen, "cpu", batch, block) for _ in range(n)]
