"""Section 5, the library way: stable-worldmodel (swm) 0.1.1 checks the world, the losses and the planner.

1. World: our TwoRoom port against swm's ``TwoRoomEnv`` (default settings),
   step for step on 50 noisy-expert episodes; our 64-pixel frames against the
   library's 224-pixel frames shrunk with OpenCV's area filter.
2. Losses: our ``sigreg`` against swm's ``SIGReg``, and section 4's variance
   and covariance terms against swm's ``VCReg``, with the same random numbers.
3. Planning: swm's ``CEMSolver`` (300 candidates, best 30, 10 rounds, cold
   start) drives the lewm seed-0 model in swm's own ``TwoRoomEnv`` on
   section 3's 200 tasks. The model still sees our 64-pixel frame of the
   env's agent position. The solver does not clip candidates; our cost clips
   them to [-1, 1] before the model sees them, as the env does.

    uv run python src/library.py --checks          # parts 1 and 2, seconds
    uv run python src/library.py --plan            # part 3, one chunk of under 5 minutes; run until it says done
    uv run python src/library.py --summary
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import warnings

import common
import numpy as np
import torch

warnings.filterwarnings("ignore")

import cv2  # noqa: E402
from stable_worldmodel.envs.two_room.env import TwoRoomEnv  # noqa: E402
from stable_worldmodel.policy import PlanConfig  # noqa: E402
from stable_worldmodel.solver.cem import CEMSolver  # noqa: E402
from stable_worldmodel.wm.loss import SIGReg, VCReg  # noqa: E402
from gymnasium import spaces  # noqa: E402

import grid_terms  # noqa: E402
import lewm as L  # noqa: E402
import plan  # noqa: E402
import tworoom as W  # noqa: E402

OUT = common.RESULTS / "library_runs"
MODEL = "lewm_s0"


def check_world(n_episodes: int = 50, steps: int = 92, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    env = TwoRoomEnv()
    worst, n, pix = 0.0, 0, []
    for ep in range(n_episodes):
        s, g = W.sample_free(rng, 1), W.sample_free(rng, 1)
        env.reset(seed=ep, options={"state": s[0], "target_state": g[0]})
        p = s.copy()
        for k in range(steps):
            a = np.clip(W.expert_action(p, g) + 0.5 * rng.standard_normal((1, 2)), -1, 1).astype(np.float32)
            obs, *_ = env.step(a[0])
            p = W.step(p, a)
            worst = max(worst, float(np.abs(obs[:2].numpy() - p[0]).max()))
            n += 1
            if k % 23 == 0:
                lib = cv2.resize(env.render(), (W.PX, W.PX), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
                ours = W.render(p)[0].permute(1, 2, 0).numpy()
                pix.append(np.abs(lib - ours))
    pix = np.stack(pix)
    return {"episodes": n_episodes, "steps": n, "max_position_diff": worst,
            "frames_compared": len(pix), "pixel_abs_diff_mean": round(float(pix.mean()), 4),
            "pixel_abs_diff_max": round(float(pix.max()), 4),
            "pixel_share_diff_over_0.25": round(float((pix.max(-1) > 0.25).mean()), 4)}


def check_losses(n: int = 50) -> dict:
    sig, std, cov = [], [], []
    lib_sig, lib_vc = SIGReg(), VCReg()
    for i in range(n):
        g = torch.Generator().manual_seed(i)
        z = torch.randn(3, 128, 32, generator=g) * (0.5 + i / n) + 0.3 * torch.randn(32, generator=g)
        torch.manual_seed(100 + i)
        a = float(lib_sig(z))
        torch.manual_seed(100 + i)
        A = torch.randn(32, L.M_PROJ)
        b = float(L.sigreg(z, A / A.norm(dim=0)))
        sig.append(abs(a - b) / abs(a))
        y = z[0]
        lv = lib_vc(y.unsqueeze(1))
        ov, oc = grid_terms.vc_terms(y)
        std.append(abs(float(lv["std_loss"]) - float(ov)) / max(abs(float(lv["std_loss"])), 1e-12))
        cov.append(abs(float(lv["cov_loss"]) - float(oc)) / abs(float(lv["cov_loss"])))
    return {"batches": n, "sigreg_max_rel_diff": float(max(sig)), "vc_std_max_rel_diff": float(max(std)),
            "vc_cov_max_rel_diff": float(max(cov))}


class OurModelAsCostable:
    """swm's Costable protocol around our model: get_cost(info_dict, candidates (B, S, H, 10)) -> (B, S)."""

    def __init__(self, latent: plan.Latent):
        self.latent = latent

    def get_cost(self, info: dict, cand: torch.Tensor) -> torch.Tensor:
        if "z0" not in info:
            info["z0"] = self.latent.encode(info["pos"][0, 0].numpy())
            info["zg"] = self.latent.encode(info["goal"][0, 0].numpy())
        self.latent.z0, self.latent.zg = info["z0"], info["zg"]
        return self.latent.cost(cand[0].clamp(-1, 1)).unsqueeze(0)


def run_task(task: dict, cost: OurModelAsCostable, seed: int) -> dict:
    env = TwoRoomEnv()
    env.reset(seed=seed, options={"state": np.array(task["start"], np.float32),
                                  "target_state": np.array(task["goal"], np.float32)})
    solver = CEMSolver(cost, num_samples=plan.SAMPLES, topk=plan.ELITES, n_steps=plan.ROUNDS, var_scale=1.0, seed=seed)
    solver.configure(action_space=spaces.Box(-1, 1, (1, 2)), n_envs=1,
                     config=PlanConfig(horizon=plan.H, receding_horizon=1, action_block=L.BLOCK, warm_start=False))
    pos = env.agent_position.numpy().copy()
    steps, done = 0, False
    for _ in range(plan.REPLANS):
        info = {"pos": torch.from_numpy(pos[None].astype(np.float32)), "goal": torch.tensor([task["goal"]], dtype=torch.float32)}
        with contextlib.redirect_stdout(io.StringIO()):                 # the solver prints its time on every call
            out = solver.solve(info)
        for a in out["actions"][0, 0].reshape(L.BLOCK, 2).numpy():
            _, _, term, _, _ = env.step(a)
            steps += 1
            if term:
                done = True
                break
        pos = env.agent_position.numpy().copy()
        if done:
            break
    return {"success": done, "steps": steps}


def plan_chunk() -> bool:
    guard = common.Guard("library plan")
    cost = OurModelAsCostable(plan.load_world(MODEL))
    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / "cem.partial.jsonl"
    done = [json.loads(x) for x in part.read_text().splitlines()] if part.exists() else []
    have = {(d["split"], d["i"]) for d in done}
    t0 = time.perf_counter()
    finished = True
    with part.open("a") as fh:
        for split, ts in plan.tasks().items():
            for i, t in enumerate(ts):
                if (split, i) in have:
                    continue
                if time.perf_counter() - t0 > plan.CHUNK_SECONDS:
                    finished = False
                    break
                r = {"split": split, "i": i, **run_task(t, cost, 10_000 + i)}
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done.append(r)
            if not finished:
                break
    rep = guard.report()
    with (OUT / "guards.jsonl").open("a") as fh:
        fh.write(json.dumps(rep) + "\n")
    if finished:
        res = {}
        for split in ("cross", "same"):
            eps = sorted((d for d in done if d["split"] == split), key=lambda d: d["i"])
            res[split] = {"success": round(float(np.mean([e["success"] for e in eps])), 4),
                          "n_success": int(sum(e["success"] for e in eps)), "per_task": [int(e["success"]) for e in eps]}
        (OUT / "cem.json").write_text(json.dumps(res) + "\n")
        part.unlink()
        print(f"  library CEM: cross {res['cross']['success']}, same {res['same']['success']}")
    return finished


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.checks:
        guard = common.Guard("library checks")
        out = {"world": check_world()}
        guard.tick("world")
        out["losses"] = check_losses()
        guard.tick("losses")
        out["guard"] = guard.report()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "checks.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps({k: v for k, v in out.items() if k != "guard"}, indent=1))
    if args.plan:
        print("done" if plan_chunk() else "not done: run again")
    if args.summary:
        import importlib.metadata as md

        checks = json.loads((OUT / "checks.json").read_text())
        cem = json.loads((OUT / "cem.json").read_text())
        ours = json.loads((plan.OUT / f"{MODEL}__cem.json").read_text())
        g = [json.loads(x) for x in (OUT / "guards.jsonl").read_text().splitlines()]
        noclip = json.loads((plan.OUT / f"{MODEL}__cem_noclip.json").read_text())
        agree = {s: int(np.sum(np.array(cem[s]["per_task"]) == np.array(ours[s]["per_task"]))) for s in ("cross", "same")}
        agree_noclip = {s: int(np.sum(np.array(cem[s]["per_task"]) == np.array(noclip[s]["per_task"])))
                        for s in ("cross", "same")}
        common.write_results("library", {
            "package": {"stable-worldmodel": md.version("stable-worldmodel"), "opencv": md.version("opencv-python-headless")},
            **{k: v for k, v in checks.items() if k != "guard"},
            "plan": {"model": MODEL, "library_cem": {s: {k: cem[s][k] for k in ("success", "n_success")} for s in cem},
                     "our_cem": {s: {k: ours[s][k] for k in ("success", "n_success")} for s in ("cross", "same")},
                     "same_outcome_tasks": agree,
                     "followup_our_cem_noclip": {s: {k: noclip[s][k] for k in ("success", "n_success")}
                                                 for s in ("cross", "same")},
                     "same_outcome_tasks_noclip": agree_noclip},
            "guards": {"checks": checks["guard"], "plan_chunks": len(g), "plan_max_minutes": max(x["minutes"] for x in g),
                       "plan_total_minutes": round(sum(x["minutes"] for x in g), 1),
                       "plan_max_peak_rss_gb": max(x["peak_rss_gb"] for x in g)},
        })


if __name__ == "__main__":
    main()
