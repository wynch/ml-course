"""Section 3: plan in the learned latent with CEM, random shooting and gradients; compare with fixed agents.

Every planner sees two frames, the current one and the goal (the agent drawn
at the goal), and returns 5 blocks of 5 actions (125 environment units of
travel at most). Its cost is LeWM's: the squared distance between the
imagined embedding after the last block and the goal's embedding. The agent
takes the first block in the real world and plans again (model-predictive
control): 30 plans, 150 steps at most per task.

  cem      LeWM's settings: 300 candidates a round, keep the best 30, refit a
           Gaussian to them, 10 rounds; the first candidate is the current mean
  shoot    random shooting: 3,000 candidates, one round (as many rollouts as CEM)
  grad     gradient descent on the actions through the predictor: 20 starts,
           25 Adam steps each, rate 0.2 (actions = tanh of free numbers), keep the best

Fixed agents: the noise-free expert (the ceiling), random actions (the floor),
CEM with the true world as its model and the straight-line distance as its
cost ("perfect model"), and CEM on the untrained network ("random weights").

Tasks: 100 with the goal in the other room (seed 3), 100 in the same room
(seed 4); start and goal at least 60 units apart along the walk.

    uv run python src/plan.py --model lewm_s0 --planner cem    # one chunk
    uv run python src/plan.py --all                            # every chunk, one process each
    uv run python src/plan.py --followup                       # written after the results: cem_best, cem_noclip
    uv run python src/plan.py --summary --figures
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import common
import numpy as np
import torch

import lewm as L
import tworoom as W

H, SAMPLES, ELITES, ROUNDS = 5, 300, 30, 10
SHOOT_SAMPLES = SAMPLES * ROUNDS
GRAD_STARTS, GRAD_STEPS, GRAD_LR = 20, 25, 0.2
CHUNK_SECONDS = 270
REPLANS = 30
BUDGET = REPLANS * L.BLOCK
N_TASKS, CROSS_SEED, SAME_SEED = 100, 3, 4
KEEP = 10                       # episodes per split kept in full for the figure and the explorable
OUT = common.RESULTS / "plan_runs"


def tasks() -> dict:
    return {"cross": W.test_tasks(N_TASKS, CROSS_SEED, cross=True), "same": W.test_tasks(N_TASKS, SAME_SEED, cross=False)}


# ------------------------------------------------------------------ costs
class Latent:
    """The learned model as a cost: encode once, imagine candidate blocks, compare with the goal."""

    def __init__(self, model: L.WorldModel, probe: np.ndarray | None = None):
        self.model = model.eval()
        self.probe = probe

    @torch.no_grad()
    def encode(self, pos) -> torch.Tensor:
        return self.model.encoder(W.render(np.asarray(pos, np.float32).reshape(1, 2)))[0]

    def start(self, pos, goal):
        self.z0, self.zg = self.encode(pos), self.encode(goal)

    def cost(self, cand: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """cand (N, H, 10) -> (N,) squared distance of the last imagined embedding to the goal's."""
        with torch.set_grad_enabled(grad):
            z = L.rollout(self.model, self.z0.expand(len(cand), -1), cand)
            return ((z[:, -1] - self.zg) ** 2).sum(-1)

    @torch.no_grad()
    def imagine_xy(self, cand: torch.Tensor) -> np.ndarray:
        """The probe's reading of each imagined step: (N, H, 2) positions; for figures only."""
        z = L.rollout(self.model, self.z0.expand(len(cand), -1), cand).numpy().astype(np.float64)
        return L.apply(self.probe, z.reshape(-1, z.shape[-1])).reshape(len(cand), H, 2)


class TrueWorld:
    """The world itself as the model, the straight-line distance as the cost."""

    def start(self, pos, goal):
        self.p0, self.g = np.asarray(pos, np.float32), np.asarray(goal, np.float32)

    def roll(self, cand: torch.Tensor) -> np.ndarray:
        a = cand.numpy().reshape(len(cand), H * L.BLOCK, 2)
        p = np.repeat(self.p0[None], len(cand), 0)
        out = []
        for k in range(H * L.BLOCK):
            p = W.step(p, a[:, k])
            if (k + 1) % L.BLOCK == 0:
                out.append(p)
        return np.stack(out, 1)

    def cost(self, cand, grad=False):
        return torch.from_numpy(((self.roll(cand)[:, -1] - self.g) ** 2).sum(-1).astype(np.float32))

    def imagine_xy(self, cand):
        return self.roll(cand)


# ---------------------------------------------------------------- planners
def cem(world, gen: torch.Generator, keep_rounds: bool = False):
    """LeWM's CEM. Returns the plan (H, 10) and, if asked, each round's best 30 imagined paths."""
    mean, std = torch.zeros(H, L.ACT_IN), torch.ones(H, L.ACT_IN)
    rounds = []
    for _ in range(ROUNDS):
        cand = (torch.randn(SAMPLES, H, L.ACT_IN, generator=gen) * std + mean).clamp(-1, 1)
        cand[0] = mean.clamp(-1, 1)
        cost = world.cost(cand)
        idx = torch.topk(cost, ELITES, largest=False).indices
        elite = cand[idx]
        if keep_rounds:
            rounds.append(np.round(world.imagine_xy(elite[:12]), 1).tolist())
        mean, std = elite.mean(0), elite.std(0)
    return mean.clamp(-1, 1), rounds


def cem_best(world, gen: torch.Generator, keep_rounds: bool = False):
    """Follow-up, written after the results: CEM as above, but return the cheapest plan seen in any
    round instead of the final mean. Same random numbers, same imagined plans."""
    mean, std = torch.zeros(H, L.ACT_IN), torch.ones(H, L.ACT_IN)
    best, best_cost = None, float("inf")
    for _ in range(ROUNDS):
        cand = (torch.randn(SAMPLES, H, L.ACT_IN, generator=gen) * std + mean).clamp(-1, 1)
        cand[0] = mean.clamp(-1, 1)
        cost = world.cost(cand)
        idx = torch.topk(cost, ELITES, largest=False).indices
        if float(cost[idx[0]]) < best_cost:
            best, best_cost = cand[idx[0]].clone(), float(cost[idx[0]])
        elite = cand[idx]
        mean, std = elite.mean(0), elite.std(0)
    return best, []


def cem_noclip(world, gen: torch.Generator, keep_rounds: bool = False):
    """Follow-up, written after section 5's result: CEM as stable-worldmodel's CEMSolver runs it. Candidates
    are clipped only on their way into the model; the Gaussian is refitted to the unclipped elites."""
    mean, std = torch.zeros(H, L.ACT_IN), torch.ones(H, L.ACT_IN)
    for _ in range(ROUNDS):
        cand = torch.randn(SAMPLES, H, L.ACT_IN, generator=gen) * std + mean
        cand[0] = mean
        elite = cand[torch.topk(world.cost(cand.clamp(-1, 1)), ELITES, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
    return mean.clamp(-1, 1), []


def shoot(world, gen: torch.Generator, keep_rounds: bool = False):
    cand = torch.randn(SHOOT_SAMPLES, H, L.ACT_IN, generator=gen).clamp(-1, 1)
    return cand[int(world.cost(cand).argmin())], []


def grad(world, gen: torch.Generator, keep_rounds: bool = False):
    u = torch.randn(GRAD_STARTS, H, L.ACT_IN, generator=gen).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=GRAD_LR)
    for _ in range(GRAD_STEPS):
        c = world.cost(torch.tanh(u), grad=True)
        opt.zero_grad()
        c.sum().backward()
        opt.step()
    with torch.no_grad():
        a = torch.tanh(u)
        return a[int(world.cost(a).argmin())], []


PLAN = {"cem": cem, "shoot": shoot, "grad": grad, "cem_best": cem_best, "cem_noclip": cem_noclip}


def run_task(task: dict, agent: str, world=None, planner: str = "cem", seed: int = 0, keep: bool = False) -> dict:
    """One task. agent: 'model' (a planner on `world`), 'random' or 'expert'."""
    if agent == "expert":
        ep = W.run_expert(task["start"], task["goal"], BUDGET)
        ep["final_dist"] = round(float(np.hypot(*(np.array(ep["path"][-1]) - np.array(task["goal"])))), 2)
        if not keep:
            del ep["path"]
        return ep
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    pos, goal = np.array([task["start"]], np.float32), np.array([task["goal"]], np.float32)
    path, imagined, first_rounds = [pos[0].tolist()], [], []
    done = False
    for r in range(REPLANS):
        if agent == "random":
            block = rng.uniform(-1, 1, (L.BLOCK, 2)).astype(np.float32)
        else:
            world.start(pos[0], goal[0])
            plan, rounds = PLAN[planner](world, gen, keep_rounds=keep and r == 0)
            if keep:
                imagined.append(np.round(world.imagine_xy(plan[None])[0], 1).tolist())
                if r == 0:
                    first_rounds = rounds
            block = plan[0].numpy().reshape(L.BLOCK, 2)
        for a in block:
            pos = W.step(pos, a[None])
            path.append(np.round(pos[0], 2).tolist())
            if np.hypot(*(pos[0] - goal[0])) < W.REACH:
                done = True
                break
        if done:
            break
    out = {"success": done, "steps": len(path) - 1, "final_dist": round(float(np.hypot(*(pos[0] - goal[0]))), 2)}
    if keep:
        out.update(path=path, imagined=imagined, first_rounds=first_rounds)
    return out


def load_world(name: str):
    """'perfect', 'random_weights', or a run name like 'lewm_s0' (needs artifacts/runs/<name>.pt from train.py)."""
    if name == "perfect":
        return TrueWorld()
    if name == "random_weights":
        import train as T

        torch.manual_seed(0)
        m = L.recalibrate(L.WorldModel(32), T.data()[0])
        return Latent(m, np.array(common.read_results("train")["random_weights"]["probe_weights"]))
    ck = torch.load(common.ARTIFACTS / "runs" / f"{name}.pt", weights_only=False, map_location="cpu")
    m = L.WorldModel(ck["dim"])
    m.load_state_dict(ck["state_eval"])
    probe = json.loads((common.RESULTS / "runs" / f"{name}.json").read_text())["probe_weights"]
    return Latent(m, np.array(probe))


def log_guard(rep: dict) -> None:
    with (OUT / "guards.jsonl").open("a") as fh:
        fh.write(json.dumps(rep) + "\n")


def chunk(model: str, planner: str) -> bool:
    """Run tasks for one (model, planner) until done or CHUNK_SECONDS pass; tasks done so far sit in a
    .partial.jsonl file, one line each. Returns True when the row is complete and written."""
    guard = common.Guard(f"plan {model} {planner}")
    world = None if model in ("random", "expert") else load_world(model)
    agent = model if model in ("random", "expert") else "model"
    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / f"{model}__{planner}.partial.jsonl"
    done = [json.loads(x) for x in part.read_text().splitlines()] if part.exists() else []
    have = {(d["split"], d["i"]) for d in done}
    t0 = time.perf_counter()
    with part.open("a") as fh:
        for split, ts in tasks().items():
            for i, t in enumerate(ts):
                if (split, i) in have:
                    continue
                if time.perf_counter() - t0 > CHUNK_SECONDS:
                    log_guard(guard.report())
                    return False
                ep = run_task(t, agent, world, planner, seed=10_000 + i, keep=i < KEEP)
                fh.write(json.dumps({"split": split, "i": i, **ep}) + "\n")
                fh.flush()
                done.append({"split": split, "i": i, **ep})
    res = {"model": model, "planner": planner}
    for split in ("cross", "same"):
        eps = sorted((d for d in done if d["split"] == split), key=lambda d: d["i"])
        succ = np.array([e["success"] for e in eps])
        res[split] = {"success": round(float(succ.mean()), 4), "n_success": int(succ.sum()),
                      "mean_steps_when_success": round(float(np.mean([e["steps"] for e in eps if e["success"]])), 2)
                      if succ.any() else None,
                      "final_dist_median": round(float(np.median([e["final_dist"] for e in eps])), 2),
                      "per_task": [int(s) for s in succ], "episodes": eps[:KEEP]}
    res["guard"] = guard.report()
    log_guard(res["guard"])
    (OUT / f"{model}__{planner}.json").write_text(json.dumps(res) + "\n")
    part.unlink()
    print(f"  {model} {planner}: cross {res['cross']['success']:.2f}, same {res['same']['success']:.2f}")
    return True


def jobs() -> list[tuple[str, str]]:
    import train as T

    out = [("expert", "none"), ("random", "none"), ("perfect", "cem"), ("random_weights", "cem")]
    out += [(f"{a}_s{s}", "cem") for a in T.ARMS for s in T.SEEDS]
    out += [(f"lewm_s{s}", p) for p in ("shoot", "grad") for s in T.SEEDS]
    return out


FOLLOWUP_JOBS = [(f"{a}_s0", "cem_best") for a in ("lewm", "dim192")] + [("lewm_s0", "cem_noclip")]   # after the results


def compact() -> None:
    """Round the kept episodes' traces (path, imagined plans, CEM rounds) to whole library units, a third
    of a pixel. Successes, steps and distances are untouched; this only shrinks the files 3 to 4 times."""
    def rnd(x):
        return np.rint(np.asarray(x, dtype=float)).astype(int).tolist()

    for f in sorted(OUT.glob("*__*.json")):
        r = json.loads(f.read_text())
        for s in ("cross", "same"):
            for e in r[s]["episodes"]:
                for k in ("path", "imagined", "first_rounds"):
                    if k in e and e[k]:
                        e[k] = rnd(e[k])
        f.write_text(json.dumps(r, separators=(",", ":")) + "\n")


def summary() -> dict:
    rows, follow = {}, {}
    for f in sorted(OUT.glob("*__*.json")):
        r = json.loads(f.read_text())
        (follow if r["planner"] in ("cem_best", "cem_noclip") else rows)[f"{r['model']}__{r['planner']}"] = {s: {k: r[s][k] for k in ("success", "n_success", "mean_steps_when_success",
                                                                         "final_dist_median", "per_task")}
                                                 for s in ("cross", "same")}
    ts = tasks()
    return {"settings": {"horizon_blocks": H, "block": L.BLOCK, "samples": SAMPLES, "elites": ELITES, "rounds": ROUNDS,
                         "shoot_samples": SHOOT_SAMPLES, "grad_starts": GRAD_STARTS, "grad_steps": GRAD_STEPS,
                         "grad_lr": GRAD_LR, "replans": REPLANS, "budget_steps": BUDGET, "reach": W.REACH},
            "tasks": {s: {"n": len(t), "geodesic_mean": round(float(np.mean([x["geodesic"] for x in t])), 2),
                          "geodesic_max": round(float(np.max([x["geodesic"] for x in t])), 2),
                          "drawn": t[-1]["drawn"], "first": t[:KEEP]} for s, t in ts.items()},
            "rows": rows,
            "followup_rows": {"written": "after the results; not part of the expectations", **follow},
            "chunk_guards": chunk_guards()}


def chunk_guards() -> dict:
    g = [json.loads(x) for x in (OUT / "guards.jsonl").read_text().splitlines()]
    return {"n": len(g), "max_minutes": max(x["minutes"] for x in g), "max_peak_rss_gb": max(x["peak_rss_gb"] for x in g),
            "total_minutes": round(sum(x["minutes"] for x in g), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--planner", default="cem")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--followup", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.all:
        for m, p in jobs():
            while not (OUT / f"{m}__{p}.json").exists():
                subprocess.run([sys.executable, __file__, "--model", m, "--planner", p], check=True)
        return
    if args.followup:
        for m, p in FOLLOWUP_JOBS:
            while not (OUT / f"{m}__{p}.json").exists():
                subprocess.run([sys.executable, __file__, "--model", m, "--planner", p], check=True)
        return
    if args.model:
        chunk(args.model, args.planner)
        return
    if args.summary:
        compact()
        guard = common.Guard("plan summary")
        S = summary()
        S["guard"] = guard.report()
        common.write_results("plan", S)
        if args.figures:
            import figures

            figures.plan()


if __name__ == "__main__":
    main()
