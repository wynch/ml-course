"""Section 1: what each rule keeps in memory while it trains.

    uv run python src/memory.py            # every rule at 2, 4, 8 and 16 hidden layers; results/memory.json
    uv run python src/memory.py --figures  # the same, then redraws figures/memory.png

Each measurement runs in its own fresh process. The process builds a wider
copy of section 2's MLP (hidden layers of 1,024, a batch of 2,048 random
images, so that the activations are big enough to see) and takes three
training steps. Two meters watch it:

- **live tensor bytes**: a dispatch mode sees every tensor torch creates,
  from the first weight to the end of the last step, and keeps a running
  total of the bytes still alive. Its peak is what the rule needs: weights,
  optimizer state, gradients and activations, nothing else. It is the CPU
  version of ``torch.cuda.max_memory_allocated``.
- **peak resident memory** of the process (``getrusage``): what the
  operating system saw, including Python, torch itself and memory the C
  allocator kept after torch freed it.

An "inference" process at the same depth builds the weights and the batch
and runs one forward pass with no gradient: the floor every rule pays. A
rule's **extra** memory is its peak minus that floor.

For backprop and checkpointing the script also counts, with autograd's
saved-tensor hooks, the bytes autograd keeps for the backward pass.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import weakref

import common
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only

import rules as R

WIDTH, BATCH, DEPTHS = 1024, 2048, (2, 4, 8, 16)
ORDER = ("inference", "backprop", "checkpointing", "mezo", "forward_forward", "mono_forward", "predictive_coding",
         "backprop_sgd", "mono_forward_sgd")
# The two _sgd rows were added after the first measurement, to separate what
# Adam's buffers cost from what the learning rule costs; they were not part of
# the expectations written before it.


class LiveBytes(TorchDispatchMode):
    """Running total of the bytes held by tensors created while the mode is on.

    Every op's outputs pass through ``__torch_dispatch__``. A storage counts
    once, however many tensors view it, and leaves the total when the last of
    those tensors is garbage-collected. Backward ops and checkpoint
    recomputation go through the same dispatcher, so they are counted too."""

    def __init__(self):
        super().__init__()
        self.refs: dict[int, int] = {}
        self.size: dict[int, int] = {}
        self.live = 0
        self.peak = 0

    def _track(self, t: torch.Tensor) -> None:
        st = t.untyped_storage()
        ptr = st.data_ptr()
        if ptr == 0:
            return
        if ptr not in self.refs:
            self.refs[ptr], self.size[ptr] = 0, st.nbytes()
            self.live += st.nbytes()
            self.peak = max(self.peak, self.live)
        self.refs[ptr] += 1
        weakref.finalize(t, self._drop, ptr)

    def _drop(self, ptr: int) -> None:
        self.refs[ptr] -= 1
        if self.refs[ptr] == 0:
            self.live -= self.size.pop(ptr)
            del self.refs[ptr]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        tree_map_only(torch.Tensor, self._track, out)
        return out


def sizes(depth: int) -> tuple[int, ...]:
    return (784,) + (WIDTH,) * depth + (10,)


def build(rule: str, depth: int):
    s = sizes(depth)
    if rule == "backprop":
        return R.Backprop(s, 0)
    if rule == "backprop_sgd":
        return R.Backprop(s, 0, opt="sgd")
    if rule == "mono_forward_sgd":
        return R.MonoForward(s, 0, opt="sgd")
    if rule == "checkpointing":
        return R.Backprop(s, 0, segments=max(1, round(math.sqrt(depth))))
    return {"inference": R.MeZO, "mezo": R.MeZO, "forward_forward": R.ForwardForward, "mono_forward": R.MonoForward,
            "predictive_coding": R.PredictiveCoding}[rule](s, 0)


def saved_bytes(learner, x, y) -> int:
    """Bytes autograd keeps for the backward pass of one step (distinct storages).
    Plain backprop's count includes the weights, which the linear layers save
    for their input gradients; under checkpointing only the segment inputs are
    saved, so its count has no weights in it."""
    seen: dict[int, int] = {}

    def pack(t):
        st = t.untyped_storage()
        seen[st.data_ptr()] = st.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = torch.nn.functional.cross_entropy(learner.forward(x), y)
    del loss
    return sum(seen.values())


def child(rule: str, depth: int) -> None:
    meter = LiveBytes()
    with meter:
        g = torch.Generator().manual_seed(0)
        x = torch.rand(BATCH, 784, generator=g)
        y = torch.randint(10, (BATCH,), generator=g)
        learner = build(rule, depth)
        out = {"rule": rule, "depth": depth}
        t0 = time.perf_counter()
        if rule == "inference":
            learner.predict(x)
        else:
            for _ in range(3):
                learner.step(x, y)
        secs = time.perf_counter() - t0
    out["seconds_per_step_metered"] = round(secs / (1 if rule == "inference" else 3), 3)
    out["live_peak_bytes"] = meter.peak
    out["peak_rss_bytes"] = common.Guard.rss_peak_bytes()
    if rule in ("backprop", "checkpointing", "backprop_sgd"):
        out["autograd_saved_bytes"] = saved_bytes(learner, x, y)
    print(json.dumps(out))


def main() -> dict:
    guard = common.Guard("memory")
    rows = []
    for depth in DEPTHS:
        for rule in ORDER:
            p = subprocess.run([sys.executable, __file__, "--child", rule, str(depth)],
                               capture_output=True, text=True, check=True)
            rows.append(json.loads(p.stdout.strip().splitlines()[-1]))
            print(f"  depth {depth:2d} {rule:18s} live {rows[-1]['live_peak_bytes'] / 2**20:7.1f} MB, "
                  f"RSS {rows[-1]['peak_rss_bytes'] / 2**20:7.1f} MB", flush=True)
            guard.tick()
    mb = lambda b: round(b / 2**20, 1)
    table = {}
    for depth in DEPTHS:
        base = next(r for r in rows if r["depth"] == depth and r["rule"] == "inference")
        n_params = sum(a * b + b for a, b in zip(sizes(depth)[:-1], sizes(depth)[1:]))
        table[str(depth)] = {"params": n_params, "params_mb": mb(4 * n_params),
                             "activation_mb_one_layer": mb(4 * BATCH * WIDTH), "rules": {}}
        for r in rows:
            if r["depth"] != depth:
                continue
            e = {"live_peak_mb": mb(r["live_peak_bytes"]), "extra_mb": mb(r["live_peak_bytes"] - base["live_peak_bytes"]),
                 "rss_peak_mb": mb(r["peak_rss_bytes"]), "rss_extra_mb": mb(r["peak_rss_bytes"] - base["peak_rss_bytes"]),
                 "seconds_per_step_metered": r["seconds_per_step_metered"]}
            if "autograd_saved_bytes" in r:
                e["autograd_saved_mb"] = mb(r["autograd_saved_bytes"])
            table[str(depth)]["rules"][r["rule"]] = e
    deep = table[str(DEPTHS[-1])]["rules"]
    out = {"width": WIDTH, "batch": BATCH, "depths": list(DEPTHS), "steps": 3,
           "checkpoint_segments": {str(d): max(1, round(math.sqrt(d))) for d in DEPTHS},
           "note": "live = peak bytes of tensors alive (weights, optimizer, gradients, activations, data); "
                   "rss = peak resident memory of the fresh process; extra = minus the inference process at the same depth",
           "by_depth": table,
           "share_of_backprop_extra_at_16": {k: round(v["extra_mb"] / deep["backprop"]["extra_mb"], 3)
                                             for k, v in deep.items() if k != "inference"}}
    out["guard"] = guard.report()
    out["guard"]["children_peak_rss_gb"] = round(max(r["peak_rss_bytes"] for r in rows) / 2**30, 2)
    common.write_results("memory", out)
    return out


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        child(sys.argv[i + 1], int(sys.argv[i + 2]))
    else:
        main()
        if "--figures" in sys.argv:
            import figures
            figures.memory()
