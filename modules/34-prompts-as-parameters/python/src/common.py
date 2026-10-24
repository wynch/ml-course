"""Shared helpers for module 34: paths, the laptop guard, results I/O, the two models.

Both models run on MLX. MLX has no memory cap like torch's MPS watermark, so
the guard only measures: wall time, peak process RSS and MLX's peak Metal
memory. Long runs stop after CHUNK_MINUTES and pick up from their log.
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2]          # modules/34-prompts-as-parameters
FIGURES = MODULE / "figures"
RESULTS = MODULE / "results"
RUNS = RESULTS / "runs"

# The frozen model whose prompt we tune: the "network".
TASK_ID = "mlx-community/Qwen3-1.7B-4bit"
TASK_REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"
# The model that proposes and edits prompts: the "optimizer".
OPT_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
OPT_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"

CHUNK_MINUTES = 3.5             # a step can take a minute; chunks stay under five


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_results(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def write_results(name: str, data: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(MODULE) if path.is_relative_to(MODULE) else path}")
    return path


class Guard:
    """Wall time and peak memory for one script or chunk."""

    def __init__(self, name: str):
        self.name = name
        self.t0 = time.perf_counter()

    def minutes(self) -> float:
        return (time.perf_counter() - self.t0) / 60

    def report(self) -> dict:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = r if sys.platform == "darwin" else r * 1024
        metal = 0
        if "mlx.core" in sys.modules:
            metal = sys.modules["mlx.core"].get_peak_memory()
        out = {
            "script": self.name,
            "minutes": round(self.minutes(), 2),
            "peak_rss_gb": round(rss / 2**30, 2),
            "peak_metal_gb": round(metal / 2**30, 2),
            "machine": f"{platform.machine()} {platform.mac_ver()[0] or platform.system()}",
        }
        print(f"[guard] {self.name}: {out['minutes']} min, peak RSS {out['peak_rss_gb']} GB, "
              f"peak Metal {out['peak_metal_gb']} GB")
        return out


def limit_cache():
    """Cap MLX's buffer cache. Without a cap, one APE run grew to 12 GB of
    resident memory in 20 minutes (every new prompt length keeps its buffers),
    and the Mac swapped."""
    import mlx.core as mx

    mx.set_cache_limit(512 * 2**20)


class Optimizer:
    """The LLM that writes and edits instructions. Loaded once per process."""

    def __init__(self, temp: float = 1.0, top_p: float = 0.95, max_tokens: int = 300):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model, self.tok = load(OPT_ID, revision=OPT_REVISION)
        limit_cache()
        self.sampler = make_sampler(temp=temp, top_p=top_p)
        self.max_tokens = max_tokens
        self.settings = {"model": f"{OPT_ID}@{OPT_REVISION}", "temp": temp,
                         "top_p": top_p, "max_tokens": max_tokens}

    def sample(self, prompt: str, n: int, seed: int) -> list[str]:
        """n independent replies to one prompt, drawn in one batch."""
        import mlx.core as mx
        from mlx_lm.generate import batch_generate

        mx.random.seed(seed)
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
        res = batch_generate(self.model, self.tok, [ids] * n, max_tokens=self.max_tokens,
                             sampler=self.sampler)
        mx.clear_cache()
        return list(res.texts)
