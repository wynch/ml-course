"""Section 4: module 25's grid JEPA, with PLDM's and LeWM's extra terms added one at a time.

Module 25's JEPA (EMA target, a predictor that outputs z plus a change)
learned a latent that drops the agent: its planner reached 3.7% of goals, as
often as random actions. Each arm here changes one thing and keeps the rest:

  ema           module 25 exactly (its own training function), the reference
  ema_plain     the predictor outputs the next latent itself, not z plus a change
  ema_idm       + inverse dynamics: a head names the action from (z_t, z_t+1)
  vc            no EMA (target = the same encoder, gradient on both sides) + variance and covariance terms
  vc_idm        vc + inverse dynamics
  pldm          vc + inverse dynamics + temporal smoothness: PLDM's full set of terms
  sigreg        no EMA + SIGReg (LeWM's single extra term)
  sigreg_idm    sigreg + inverse dynamics

Weights, fixed before any run: variance 1, covariance 0.04 (VICReg's 25 : 25 : 1
ratio), inverse dynamics 1, temporal smoothness 0.1, SIGReg 0.1 (256 directions,
as LeJEPA's code default, to keep each arm under 5 minutes). Three seeds per
arm; the data, the 300 test arenas, the probe and the random-shooting planner
are module 25's.

    uv run python src/grid_terms.py --arm vc       # one arm, three seeds (well under 5 minutes)
    uv run python src/grid_terms.py --all          # every arm, one process each
    uv run python src/grid_terms.py --summary --figures
    uv run python src/grid_terms.py --followup all       # written after the results: see FOLLOW_ARMS
    uv run python src/grid_terms.py --followup summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import common
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import grid
import grid_models as G
import grid_plan as GP
import lewm

ARMS = {
    "ema": {}, "ema_plain": {"plain": True}, "ema_idm": {"idm": 1.0},
    "vc": {"no_ema": True, "var": 1.0, "cov": 0.04}, "vc_idm": {"no_ema": True, "var": 1.0, "cov": 0.04, "idm": 1.0},
    "pldm": {"no_ema": True, "var": 1.0, "cov": 0.04, "idm": 1.0, "smooth": 0.1},
    "sigreg": {"no_ema": True, "sigreg": 0.1}, "sigreg_idm": {"no_ema": True, "sigreg": 0.1, "idm": 1.0},
}
SEEDS = (0, 1, 2)
OUT = common.RESULTS / "grid_runs"
TRAIN_SEED, TRAIN_ARENAS, HELD_SEED, HELD_ARENAS, WALK = 1, 4000, 2, 500, 25     # module 25's
TEST_SEED, N_TEST = 3, 300


class PlainPredictor(nn.Module):
    """z' = f(z, a): staying put has to be learned, it is not the default."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(G.LATENT + 4, G.HIDDEN), nn.ReLU(), nn.Linear(G.HIDDEN, G.LATENT))

    def forward(self, z, action):
        return self.net(torch.cat([z, F.one_hot(action, 4).float()], -1))


def vc_terms(z: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg's variance hinge (each latent's std over the batch pushed up to 1) and covariance
    term (off-diagonal squared covariances, divided by D^2 - D, as stable-worldmodel's VCReg)."""
    z = z - z.mean(0)
    std = (z.var(0) + eps).sqrt()
    D = z.shape[1]
    cov = z.T @ z / (len(z) - 1)
    off = (cov ** 2).sum() - (cov.diagonal() ** 2).sum()
    return F.relu(1 - std).mean(), off / (D * D - D)


def make_model(cfg: dict) -> tuple[G.JEPA, nn.Module | None]:
    m = G.JEPA()
    if cfg.get("plain"):
        m.predictor = PlainPredictor()
    if cfg.get("no_ema"):
        m.target = m.encoder                     # one encoder: the target carries gradient
    idm = nn.Sequential(nn.Linear(2 * G.LATENT, G.HIDDEN), nn.ReLU(), nn.Linear(G.HIDDEN, 4)) if cfg.get("idm") else None
    return m, idm


def loss_terms(m: G.JEPA, idm, cfg: dict, obs, action, next_obs) -> dict:
    """Every term of one arm on one batch; 'total' is what the optimiser sees."""
    z = m.encoder(obs)
    pred = m.predictor(z, action)
    if cfg.get("no_ema"):
        tgt = m.encoder(next_obs)
    else:
        with torch.no_grad():
            tgt = m.target(next_obs)
    out = {"pred": ((pred - tgt) ** 2).mean()}
    z_next = tgt if cfg.get("no_ema") else m.encoder(next_obs)
    both = torch.cat([z, z_next])
    if cfg.get("var"):
        out["var"], out["cov"] = vc_terms(both)
    if cfg.get("idm"):
        out["idm"] = F.cross_entropy(idm(torch.cat([z, z_next], -1)), action)
    if cfg.get("smooth"):
        out["smooth"] = ((z - z_next) ** 2).mean()
    if cfg.get("sigreg"):
        out["sigreg"] = lewm.sigreg(both, M=256)
    weight = {"pred": 1.0, "var": cfg.get("var", 0), "cov": cfg.get("cov", 0), "idm": cfg.get("idm", 0),
              "smooth": cfg.get("smooth", 0), "sigreg": cfg.get("sigreg", 0)}
    out["total"] = sum(weight[k] * v for k, v in out.items())
    return out


def train(cfg: dict, data: dict, seed: int) -> tuple[G.JEPA, list]:
    """Module 25's loop (same steps, batch, rate, sampling, EMA ramp), with this arm's terms."""
    if not cfg:
        return G.train_jepa(data, seed=seed)                  # module 25's own function, untouched
    torch.manual_seed(seed)
    m, idm = make_model(cfg)
    params = list(m.encoder.parameters()) + list(m.predictor.parameters()) + (list(idm.parameters()) if idm else [])
    opt = torch.optim.Adam(params, lr=G.LR)
    T = G.tensors(data)
    g = torch.Generator().manual_seed(seed)
    log = []
    for s in range(1, G.STEPS + 1):
        i = torch.randint(len(T["obs"]), (G.BATCH,), generator=g)
        out = loss_terms(m, idm, cfg, T["obs"][i], T["action"][i], T["next_obs"][i])
        opt.zero_grad()
        out["total"].backward()
        opt.step()
        if not cfg.get("no_ema"):
            G.ema_update(m.target, m.encoder, G.ema_momentum(s, G.STEPS))
        if s % 200 == 0:
            log.append({"step": s, **{k: round(v.item(), 6) for k, v in out.items()}})
    return m.eval(), log


@torch.no_grad()
def read_out(m: G.JEPA, data: dict, held: dict, arenas: list) -> dict:
    """Module 25's checks: held-out loss, what a ridge probe reads, the collapse meters, planning."""
    H = G.tensors(held)
    loss = float(G.jepa_loss(m, H["obs"], H["action"], H["next_obs"]))
    n = 20_000
    tr = {k: v[:n] for k, v in data.items()}
    labels = {"agent_cell": lambda d: d["pos"], "goal_offset": lambda d: d["goal"] - d["pos"],
              "wall_count": lambda d: d["obs"][:, :grid.N * grid.N].sum(1, keepdims=True)}
    Ztr, Zte = G.target_latents(m, tr["obs"]), G.target_latents(m, held["obs"])
    probe_r2 = {k: round(G.r2(G.Probe.fit(Ztr, f(tr))(Zte), f(held)), 4) for k, f in labels.items()}
    moved = ~(held["pos"] == held["next_pos"]).all(-1)
    z2 = G.target_latents(m, held["next_obs"])
    rng = np.random.default_rng(0)
    i, j = rng.integers(len(Zte), size=(2, 5000))
    other = held["arena"][i] != held["arena"][j]
    probe = G.fit_probe(m, data)
    res = GP.evaluate({"latent": m}, {"latent": probe}, arenas, keep=4, agents=("latent",))["latent"]
    return {"held_pred_loss": round(loss, 6), "probe_r2": probe_r2, "meters": G.collapse_meters(Zte[:2048]),
            "move_one_step": round(float(np.linalg.norm(z2 - Zte, axis=1)[moved].mean()), 5),
            "between_arenas": round(float(np.linalg.norm(Zte[i] - Zte[j], axis=1)[other].mean()), 5),
            "plan_success": res["success"], "plan_n_success": res["n_success"],
            "plan_by_distance": res["by_distance"],
            "episodes": [{"success": e["success"], "steps": e["steps"], "path": e["path"]} for e in res["episodes"]]}


def run_arm(arm: str) -> dict:
    guard = common.Guard(f"grid {arm}")
    data = grid.collect(TRAIN_ARENAS, WALK, TRAIN_SEED)
    held = grid.collect(HELD_ARENAS, WALK, HELD_SEED)
    arenas = grid.test_arenas(N_TEST, TEST_SEED, GP.MIN_DIST, GP.MAX_DIST)
    out = {"arm": arm, "cfg": ARMS[arm], "seeds": {}}
    for seed in SEEDS:
        m, log = train(ARMS[arm], data, seed)
        guard.tick(f"train_s{seed}")
        out["seeds"][str(seed)] = {**read_out(m, data, held, arenas), "log": log[::5]}
        guard.tick(f"read_s{seed}")
        print(f"  {arm} s{seed}: agent R2 {out['seeds'][str(seed)]['probe_r2']['agent_cell']}, "
              f"plan {out['seeds'][str(seed)]['plan_success']}")
    out["guard"] = guard.report()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{arm}.json").write_text(json.dumps(out) + "\n")
    return out


def random_baselines() -> dict:
    """Random actions and a perfect model on the same 300 arenas (module 25's floor and planner ceiling)."""
    arenas = grid.test_arenas(N_TEST, TEST_SEED, GP.MIN_DIST, GP.MAX_DIST)
    res = GP.evaluate({}, {}, arenas, keep=0, agents=("random", "perfect"))
    return {a: {"success": res[a]["success"], "n_success": res[a]["n_success"]} for a in res}


def summary() -> dict:
    arms = {}
    for arm in ARMS:
        r = json.loads((OUT / f"{arm}.json").read_text())
        S = r["seeds"]
        arms[arm] = {"cfg": r["cfg"], "plan_success": [S[s]["plan_success"] for s in S],
                     "agent_r2": [S[s]["probe_r2"]["agent_cell"] for s in S],
                     "goal_offset_r2": [S[s]["probe_r2"]["goal_offset"] for s in S],
                     "wall_r2": [S[s]["probe_r2"]["wall_count"] for s in S],
                     "erank": [S[s]["meters"]["erank"] for s in S], "std_norm": [S[s]["meters"]["std_norm"] for s in S],
                     "held_pred_loss": [S[s]["held_pred_loss"] for s in S],
                     "move_one_step": [S[s]["move_one_step"] for s in S],
                     "between_arenas": [S[s]["between_arenas"] for s in S],
                     "minutes": r["guard"]["minutes"], "peak_rss_gb": r["guard"]["peak_rss_gb"]}
    return {"arms": arms, "seeds": list(SEEDS), "n_test": N_TEST}


# ------------------------------------------------- follow-up, written after the results
FOLLOW_ARMS = ("ema", "ema_idm", "vc_idm", "pldm", "sigreg_idm")


class AgentProbe:
    """Reads the agent's cell from the latent and returns the goal's offset from it, for one known goal.

    Module 25's planner reads the offset straight from the latent, so the goal
    must be in the latent too. This probe asks only for the agent; the goal
    cell comes from outside, as the one-hot planner's score takes it.
    """

    def __init__(self, probe: G.Probe, goal):
        self.probe, self.goal = probe, torch.tensor(goal, dtype=torch.float32)

    def __call__(self, Z):
        return self.goal - self.probe(Z)


def followup(arm: str) -> dict:
    """Retrain the arm (CPU, same seeds: the same weights), check that it reproduces its plan success, then
    measure what the latent keeps of the goal and plan with the agent-only probe."""
    guard = common.Guard(f"grid followup {arm}")
    data = grid.collect(TRAIN_ARENAS, WALK, TRAIN_SEED)
    held = grid.collect(HELD_ARENAS, WALK, HELD_SEED)
    arenas = grid.test_arenas(N_TEST, TEST_SEED, GP.MIN_DIST, GP.MAX_DIST)
    saved = json.loads((OUT / f"{arm}.json").read_text())["seeds"]
    out = {"arm": arm, "seeds": {}}
    n = 20_000
    for seed in SEEDS:
        m, _ = train(ARMS[arm], data, seed)
        Ztr, Zte = G.target_latents(m, data["obs"][:n]), G.target_latents(m, held["obs"])
        goal_r2 = G.r2(G.Probe.fit(Ztr, data["goal"][:n])(Zte), held["goal"])
        agent_probe = G.Probe.fit(Ztr, data["pos"][:n])
        same = GP.evaluate({"latent": m}, {"latent": G.fit_probe(m, data)}, arenas, keep=0, agents=("latent",))
        wins = []
        for i, ar in enumerate(arenas):
            ep = GP.run_episode("latent", ar, np.random.default_rng(1000 + i), {"latent": m},
                                {"latent": AgentProbe(agent_probe, ar["goal"])})
            wins.append(ep["success"])
        out["seeds"][str(seed)] = {"reproduced_plan_success": same["latent"]["success"],
                                   "saved_plan_success": saved[str(seed)]["plan_success"],
                                   "goal_cell_r2": round(goal_r2, 4),
                                   "agent_probe_plan_success": round(float(np.mean(wins)), 4)}
        guard.tick(f"s{seed}")
        print(f"  {arm} s{seed}: {out['seeds'][str(seed)]}")
    out["guard"] = guard.report()
    (OUT / f"followup_{arm}.json").write_text(json.dumps(out) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--followup", choices=FOLLOW_ARMS + ("all", "summary"))
    args = ap.parse_args()
    if args.followup == "all":
        for arm in FOLLOW_ARMS:
            if not (OUT / f"followup_{arm}.json").exists():
                subprocess.run([sys.executable, __file__, "--followup", arm], check=True)
        return
    if args.followup == "summary":
        F = {a: json.loads((OUT / f"followup_{a}.json").read_text()) for a in FOLLOW_ARMS}
        common.write_results("grid_followup", {
            "written": "after the results of grid.json; not part of the expectations",
            "arms": {a: {k: [F[a]["seeds"][s][k] for s in F[a]["seeds"]] for k in F[a]["seeds"]["0"]} for a in F},
            "max_minutes": max(F[a]["guard"]["minutes"] for a in F),
            "max_peak_rss_gb": max(F[a]["guard"]["peak_rss_gb"] for a in F)})
        return
    if args.followup:
        followup(args.followup)
        return
    if args.all:
        for arm in ARMS:
            if not (OUT / f"{arm}.json").exists():
                subprocess.run([sys.executable, __file__, "--arm", arm], check=True)
        return
    if args.arm:
        run_arm(args.arm)
        return
    if args.summary:
        guard = common.Guard("grid summary")
        S = summary()
        S["baselines"] = random_baselines()
        guard.tick("baselines")
        S["guard"] = guard.report()
        common.write_results("grid", S)
        if args.figures:
            import figures

            figures.grid()


if __name__ == "__main__":
    main()
