"""Section 2: train the pixel world model, four ways, three seeds each, in chunks of under 5 minutes.

Arms (written down in results/expectations.json before any of these runs):
  lewm       D = 32,  lambda = 0.1   LeWM's recipe at small scale
  pred_only  D = 32,  lambda = 0     the prediction loss alone: the collapse control
  dim8       D = 8,   lambda = 0.1   fewer dimensions for a world with 2 degrees of freedom
  dim192     D = 192, lambda = 0.1   LeWM's own embedding size

Data: 4,000 episodes x 92 steps of a noisy expert (seed 1); 500 held-out
episodes for every number (seed 2). Each run: 3,000 Adam steps of 128
sub-trajectories (4 frames, 5 environment steps apart), rate 3e-4, on MPS.

A run saves itself to artifacts/runs/<arm>_s<seed>.pt every chunk and goes on
from there, so no process runs over CHUNK_SECONDS of training. When a run
ends, every BatchNorm's running statistics are recomputed on the final
weights (see lewm.recalibrate), and the run is read out (held-out loss,
probes, collapse meters, distances, imagined against real) into
results/runs/<arm>_s<seed>.json.

    uv run python src/train.py --arm lewm --seed 0     # one chunk of one run
    uv run python src/train.py --all                   # every chunk of every run, one process each
    uv run python src/train.py --summary --figures     # results/train.json and its figure from the run files
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

ARMS = {"lewm": {"dim": 32, "lam": 0.1}, "pred_only": {"dim": 32, "lam": 0.0},
        "dim8": {"dim": 8, "lam": 0.1}, "dim192": {"dim": 192, "lam": 0.1}}
SEEDS = (0, 1, 2)
STEPS, BATCH, LR, LOG_EVERY = 3000, 128, 3e-4, 50
TRAIN_SEED, TRAIN_EPISODES, HELD_SEED, HELD_EPISODES, EP_STEPS = 1, 4000, 2, 500, 92
CHUNK_SECONDS = 270
GEO_BINS = (0, 10, 20, 30, 45, 60, 80, 100, 130, 160, 200, 260, 400)
RUNS = common.ARTIFACTS / "runs"
RUN_RESULTS = common.RESULTS / "runs"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def data() -> tuple[dict, dict]:
    return W.collect(TRAIN_EPISODES, EP_STEPS, TRAIN_SEED), W.collect(HELD_EPISODES, EP_STEPS, HELD_SEED)


def train_chunk(arm: str, seed: int, guard: common.Guard, steps: int = STEPS, budget: float = CHUNK_SECONDS) -> bool:
    """Train one run until it ends or the chunk's time is up. Returns True when the run is done."""
    cfg, dev = ARMS[arm], device()
    path = RUNS / f"{arm}_s{seed}.pt"
    train, _ = data()
    torch.manual_seed(seed)
    model = L.WorldModel(cfg["dim"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(1000 + seed)
    state = {"step": 0, "log": [], "seconds": 0.0, "chunks": 0}
    if path.exists():
        ck = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        state = ck["progress"]
        if state["step"] >= steps:
            return True
    model.train()
    t0, acc = time.perf_counter(), {"pred": 0.0, "sigreg": 0.0}
    while state["step"] < steps and time.perf_counter() - t0 < budget:
        frames, act, _ = L.sample_batch(train, rng, BATCH)
        out = L.losses(model, frames.to(dev), act.to(dev), cfg["lam"])
        opt.zero_grad()
        out["total"].backward()
        opt.step()
        state["step"] += 1
        acc["pred"] += out["pred"].item()
        acc["sigreg"] += out["sigreg"].item()
        if state["step"] % LOG_EVERY == 0:
            state["log"].append({"step": state["step"], "pred": round(acc["pred"] / LOG_EVERY, 6),
                                 "sigreg": round(acc["sigreg"] / LOG_EVERY, 4)})
            acc = {"pred": 0.0, "sigreg": 0.0}
        guard.tick()
    state["seconds"] = round(state["seconds"] + time.perf_counter() - t0, 1)
    state["chunks"] += 1
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save({"dim": cfg["dim"], "state": model.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(), "progress": state, "arm": arm, "seed": seed}, path)
    print(f"  {arm} s{seed}: step {state['step']}/{steps}, {state['seconds']} s of training in {state['chunks']} chunks")
    return state["step"] >= steps


@torch.no_grad()
def read_out(model: L.WorldModel, train: dict, held: dict, dev: str = "cpu") -> dict:
    """Every held-out number for one trained model."""
    model = model.to(dev).eval()
    rng = np.random.default_rng(7)
    # held-out losses, on 20 batches
    lp, ls = [], []
    for _ in range(20):
        f, a, _ = L.sample_batch(held, rng, BATCH)
        out = L.losses(model, f.to(dev), a.to(dev), 1.0, gen=torch.Generator(device=dev).manual_seed(0))
        lp.append(float(out["pred"]))
        ls.append(float(out["sigreg"]))
    # probes: position (x, y) from the embedding, fitted on 20,000 training frames, scored on 5,000 held-out ones
    Ptr = train["pos"].reshape(-1, 2)[rng.choice(train["pos"].shape[0] * train["pos"].shape[1], 20_000, replace=False)]
    Pte = held["pos"].reshape(-1, 2)[rng.choice(held["pos"].shape[0] * held["pos"].shape[1], 5_000, replace=False)]
    Ztr, Zte = L.embed(model, Ptr, dev), L.embed(model, Pte, dev)
    Wp = L.ridge(Ztr, Ptr)
    pred = L.apply(Wp, Zte)
    room_tr, room_te = (Ptr[:, 0] > W.WALL_X).astype(float)[:, None], (Pte[:, 0] > W.WALL_X)
    room_acc = float(((L.apply(L.ridge(Ztr, room_tr), Zte)[:, 0] > 0.5) == room_te).mean())
    # distances: does latent distance follow the walk around the wall?
    i, j = rng.integers(len(Pte), size=(2, 5000))
    dz = np.linalg.norm(Zte[i] - Zte[j], axis=1)
    cross = (Pte[i, 0] > W.WALL_X) != (Pte[j, 0] > W.WALL_X)
    geo, euc = W.geodesic(Pte[i], Pte[j]), np.hypot(*(Pte[i] - Pte[j]).T)
    # imagined against real: 6 blocks from 1,000 held-out starts, read through the probe
    E = held["pos"].shape[0]
    e, t0 = rng.integers(E, size=1000), rng.integers(0, EP_STEPS - 6 * L.BLOCK + 1, size=1000)
    z0 = torch.from_numpy(L.embed(model, held["pos"][e, t0], dev)).float().to(dev)
    acts =np.stack([held["action"][e[n], t0[n]:t0[n] + 6 * L.BLOCK] for n in range(1000)]).reshape(1000, 6, L.ACT_IN)
    zs = L.rollout(model, z0, torch.from_numpy(acts).to(dev)).cpu().numpy().astype(np.float64)
    real = np.stack([held["pos"][e, t0 + (k + 1) * L.BLOCK] for k in range(6)], 1)
    start = held["pos"][e, t0]
    imagined = L.apply(Wp, zs.reshape(-1, zs.shape[-1])).reshape(1000, 6, 2)
    enc_real = L.apply(Wp, L.embed(model, real.reshape(-1, 2), dev)).reshape(1000, 6, 2)
    return {
        "held_pred_loss": round(float(np.mean(lp)), 6), "held_sigreg": round(float(np.mean(ls)), 4),
        "probe_r2_xy": round(L.r2(pred, Pte), 4),
        "probe_r2_x": round(L.r2(pred[:, :1], Pte[:, :1]), 4), "probe_r2_y": round(L.r2(pred[:, 1:], Pte[:, 1:]), 4),
        "probe_err_units": round(float(np.hypot(*(pred - Pte).T).mean()), 2),
        "room_acc": round(room_acc, 4),
        "meters": L.collapse_meters(Zte[:2048]),
        "latent_std_mean": round(float(Zte.std(0).mean()), 4),
        "spearman_geodesic": round(L.spearman(dz, geo), 4), "spearman_euclid": round(L.spearman(dz, euc), 4),
        "spearman_geodesic_cross": round(L.spearman(dz[cross], geo[cross]), 4),
        "spearman_euclid_cross": round(L.spearman(dz[cross], euc[cross]), 4),
        "sqdist_by_geodesic": {"edges": list(GEO_BINS),
                               "median": [round(float(np.median(dz[(geo >= a) & (geo < b)] ** 2)), 3)
                                          if ((geo >= a) & (geo < b)).any() else None
                                          for a, b in zip(GEO_BINS[:-1], GEO_BINS[1:])]},
        "rollout_err_units": [round(float(np.hypot(*(imagined[:, k] - real[:, k]).T).mean()), 2) for k in range(6)],
        "probe_of_real_err_units": [round(float(np.hypot(*(enc_real[:, k] - real[:, k]).T).mean()), 2) for k in range(6)],
        "stay_put_err_units": [round(float(np.hypot(*(start - real[:, k]).T).mean()), 2) for k in range(6)],
        "probe_weights": np.round(Wp, 6).tolist(),
    }


def finish(arm: str, seed: int) -> dict:
    """Read out a finished run, write its JSON, keep the run's weights for plan.py."""
    ck = torch.load(RUNS / f"{arm}_s{seed}.pt", weights_only=False, map_location="cpu")
    model = L.WorldModel(ck["dim"])
    model.load_state_dict(ck["state"])
    train, held = data()
    L.recalibrate(model.to(device()), train, device=device())
    ck["state_eval"] = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(ck, RUNS / f"{arm}_s{seed}.pt")
    out = {"arm": arm, "seed": seed, **ARMS[arm], "progress": ck["progress"], **read_out(model, train, held, device())}
    RUN_RESULTS.mkdir(parents=True, exist_ok=True)
    (RUN_RESULTS / f"{arm}_s{seed}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"  {arm} s{seed}: probe R2 {out['probe_r2_xy']}, erank {out['meters']['erank']}, "
          f"held pred {out['held_pred_loss']}")
    return out


def random_weights_readout() -> dict:
    """The same network at its initial weights: what the probe reads from an untrained encoder."""
    torch.manual_seed(0)
    model = L.WorldModel(32)
    train, held = data()
    L.recalibrate(model, train)
    return {"arm": "random_weights", "dim": 32, **read_out(model, train, held, "cpu")}


def summary() -> dict:
    runs = {}
    for f in sorted(RUN_RESULTS.glob("*_s*.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["arm"], []).append(r)
    keys = ("probe_r2_xy", "probe_err_units", "room_acc", "held_pred_loss", "spearman_geodesic", "spearman_euclid",
            "spearman_geodesic_cross", "spearman_euclid_cross")
    out = {"arms": {}}
    for arm, rs in runs.items():
        out["arms"][arm] = {"seeds": [r["seed"] for r in rs], **ARMS[arm],
                            **{k: [r[k] for r in rs] for k in keys},
                            "erank": [r["meters"]["erank"] for r in rs], "std_norm": [r["meters"]["std_norm"] for r in rs],
                            "rollout_err_units": [r["rollout_err_units"] for r in rs],
                            "minutes": [round(r["progress"]["seconds"] / 60, 2) for r in rs],
                            "chunks": [r["progress"]["chunks"] for r in rs]}
    return out


LOOKUP = 48          # positions per side of the encoder lookup table the explorable uses


@torch.no_grad()
def export_planner(name: str = "lewm_s0") -> dict:
    """artifacts/lewm_planner.pt: what the explorable needs from one run, small enough to commit.

    The predictor (16-bit floats, BatchNorm buffers kept 32-bit), the probe, and
    the encoder's output on a LOOKUP x LOOKUP grid of agent positions over
    [LO, HI]^2: the frame depends only on the agent's position, so the table
    stands in for the encoder. The full runs stay in artifacts/runs/, not in git.
    """
    ck = torch.load(RUNS / f"{name}.pt", weights_only=False, map_location="cpu")
    model = L.WorldModel(ck["dim"])
    model.load_state_dict(ck["state_eval"])
    model.eval()
    u = np.linspace(W.LO, W.HI, LOOKUP, dtype=np.float32)
    gy, gx = np.meshgrid(u, u, indexing="ij")
    table = L.embed(model, np.stack([gx.ravel(), gy.ravel()], 1)).astype(np.float32).reshape(LOOKUP, LOOKUP, -1)
    pred = {k: (v.half() if v.is_floating_point() and "running" not in k else v)
            for k, v in model.predictor.state_dict().items()}
    probe = json.loads((RUN_RESULTS / f"{name}.json").read_text())["probe_weights"]
    path = common.ARTIFACTS / "lewm_planner.pt"
    torch.save({"run": name, "dim": ck["dim"], "predictor": pred, "lookup": torch.from_numpy(table).half(),
                "lookup_axis": u.tolist(), "probe": probe}, path)
    return {"path": "artifacts/lewm_planner.pt", "run": name, "bytes": path.stat().st_size, "lookup": LOOKUP}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.all:
        for arm in ARMS:
            for seed in SEEDS:
                while not (RUN_RESULTS / f"{arm}_s{seed}.json").exists():
                    subprocess.run([sys.executable, __file__, "--arm", arm, "--seed", str(seed)], check=True)
        return
    if args.arm:
        guard = common.Guard(f"train {args.arm} s{args.seed}", device())
        done = train_chunk(args.arm, args.seed, guard)
        guard.tick("train")
        if done and not (RUN_RESULTS / f"{args.arm}_s{args.seed}.json").exists():
            finish(args.arm, args.seed)
            guard.tick("read_out")
        rep = guard.report()
        log = RUN_RESULTS / "guards.jsonl"
        RUN_RESULTS.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            fh.write(json.dumps(rep) + "\n")
        return
    if args.summary:
        guard = common.Guard("train summary")
        S = summary()
        S["random_weights"] = random_weights_readout()
        guard.tick("random_weights")
        train, held = data()
        S["data"] = {"train_episodes": TRAIN_EPISODES, "held_episodes": HELD_EPISODES, "steps": EP_STEPS,
                     "seeds": {"train": TRAIN_SEED, "held": HELD_SEED},
                     "crossing_share_train": round(float(((train["pos"][:, :, 0] < W.WALL_X).any(1)
                                                          & (train["pos"][:, :, 0] > W.WALL_X).any(1)).mean()), 4)}
        S["settings"] = {"steps": STEPS, "batch": BATCH, "lr": LR, "seq": L.SEQ, "block": L.BLOCK,
                         "sigreg_projections": L.M_PROJ, "knots": L.KNOTS, "device": device()}
        m = L.WorldModel(32)
        S["params"] = {"encoder": L.n_params(m.encoder), "predictor": L.n_params(m.predictor)}
        guards = [json.loads(x) for x in (RUN_RESULTS / "guards.jsonl").read_text().splitlines()]
        S["chunk_guards"] = {"n": len(guards), "max_minutes": max(g["minutes"] for g in guards),
                             "max_peak_rss_gb": max(g["peak_rss_gb"] for g in guards),
                             "max_peak_mps_driver_gb": max(g["peak_mps_driver_gb"] for g in guards)}
        S["planner_artifact"] = export_planner()
        guard.tick("artifact")
        S["guard"] = guard.report()
        common.write_results("train", S)
        if args.figures:
            import figures

            figures.train()


if __name__ == "__main__":
    main()
