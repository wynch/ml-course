"""The library way: OpenEvolve, an open re-implementation of AlphaEvolve's loop.

OpenEvolve talks to any OpenAI-compatible endpoint. No API key exists here,
so this script starts `mlx_lm.server` with the same local model as our loop
and points OpenEvolve at it. The evaluator is ours (src/oe_evaluator.py):
the same sandbox, the same evolve set, score = minus the excess.

Runs in chunks: each call runs at most LIB_CHUNK iterations, then saves a
checkpoint; run it again to continue.

    uv run python src/library.py           # until LIB_ITERATIONS are done
    uv run python src/library.py --score   # held-out score of OpenEvolve's best
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import common
import sandbox
import task

OUT = common.MODULE / "python" / "outputs" / "openevolve"
PORT = 8081
LIB_ITERATIONS = 100       # one LLM call and one evaluator call each
LIB_CHUNK = 20

INITIAL = '''import numpy as np

# EVOLVE-BLOCK-START
def priority(item, bins):
    """Score each open bin that can take the item; the highest score wins.

    item: size of the arriving item. bins: numpy array, remaining room of each
    open bin that fits it. Goal: use as few bins as possible in online bin packing.
    """
    return -(bins - item)
# EVOLVE-BLOCK-END
'''


def config_text() -> str:
    return f"""max_iterations: {LIB_ITERATIONS}
checkpoint_interval: 10
random_seed: 0
diff_based_evolution: false
max_code_length: 4000
llm:
  api_base: "http://127.0.0.1:{PORT}/v1"
  api_key: "local"
  models:
    - name: "{common.MODEL_ID}"
      weight: 1.0
  temperature: 1.0
  top_p: 0.95
  max_tokens: 1024
  timeout: 300
  retries: 2
prompt:
  system_message: "You improve a Python priority function for online bin packing. Items arrive one at a time; each goes into the open bin with the highest priority among bins with room. Fewer bins is better. Keep the function short and use only numpy."
  num_top_programs: 2
  num_diverse_programs: 1
database:
  population_size: 200
  num_islands: 4
  migration_interval: 25
  random_seed: 0
evaluator:
  timeout: 30
  cascade_evaluation: false
  parallel_evaluations: 1
"""


def server_up() -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
        return True
    except OSError:
        return False


def start_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm.server", "--model", common.MODEL_ID, "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        if server_up():
            return proc
        time.sleep(1)
    proc.kill()
    raise SystemExit("mlx_lm.server did not start")


def last_checkpoint() -> Path | None:
    cps = sorted((OUT / "checkpoints").glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))
    return cps[-1] if cps else None


def run_chunk() -> None:
    from openevolve.api import run_evolution

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "initial_program.py").write_text(INITIAL)
    (OUT / "config.yaml").write_text(config_text())
    cp = last_checkpoint()
    done = int(cp.name.split("_")[1]) if cp else 0
    if done >= LIB_ITERATIONS:
        print(f"openevolve: done ({done} iterations)")
        return
    guard = common.Guard(f"library.py from {done}")
    server = None if server_up() else start_server()
    try:
        run_evolution(str(OUT / "initial_program.py"), str(Path(__file__).with_name("oe_evaluator.py")),
                      config=str(OUT / "config.yaml"), iterations=min(LIB_CHUNK, LIB_ITERATIONS - done),
                      output_dir=str(OUT), cleanup=False, checkpoint_path=str(cp) if cp else None)
    finally:
        if server:
            server.terminate()
            server.wait()
    g = guard.report()
    cp = last_checkpoint()
    g["iterations_after"] = int(cp.name.split("_")[1]) if cp else 0
    common.append_jsonl(common.RESULTS / "library_guards.jsonl", g)


def score() -> None:
    cp = last_checkpoint()
    best_code = (cp / "best_program.py").read_text()
    m = re.search(r"# EVOLVE-BLOCK-START\n(.*?)# EVOLVE-BLOCK-END", best_code, re.S)
    code = m.group(1) if m else best_code
    out = sandbox.evaluate(code, sets=("evolve", "heldout", "shift"))
    programs = list((cp / "programs").glob("*.json"))
    valid, scores = 0, {}
    for p in programs:
        metrics = json.loads(p.read_text()).get("metrics", {})
        if metrics.get("valid", 0) == 1:
            valid += 1
            key = f"{metrics['excess_percent']:.4f}"
            scores[key] = scores.get(key, 0) + 1
    guards = common.read_jsonl(common.RESULTS / "library_guards.jsonl")
    data = {
        "library": "openevolve 0.4.0", "model": f"{common.MODEL_ID} via mlx_lm.server",
        "iterations": int(cp.name.split("_")[1]), "programs_in_database": len(programs),
        "valid_programs_in_database": valid,
        "valid_excess_counts": dict(sorted(scores.items())),
        "best_code": code, "status": out["status"],
        "excess": {s: round(task.excess(b, s), 4) for s, b in out["bins"].items()},
        "best_fit": {s: round(task.excess(task.bins_used(task.best_fit, s), s), 4) for s in task.SETS},
        "chunks": guards,
    }
    common.write_results("library", data)
    print(json.dumps(data["excess"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if args.reset:
        shutil.rmtree(OUT, ignore_errors=True)
        (common.RESULTS / "library_guards.jsonl").unlink(missing_ok=True)
    if args.score:
        score()
    else:
        run_chunk()


if __name__ == "__main__":
    main()
