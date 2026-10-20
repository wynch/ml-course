"""Redraw every figure of module 31 from results/*.json; no model is loaded.

    uv run python src/figures.py
"""

from __future__ import annotations

import json

import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tworoom as W

ARM_LABEL = {"lewm": "LeWM recipe, D = 32", "pred_only": "prediction only, D = 32", "dim8": "D = 8", "dim192": "D = 192"}
ARM_COL = {"lewm": "#1d6fb8", "pred_only": "#d1495b", "dim8": "#3fa34d", "dim192": "#7b4fb0", "random_weights": "#8a8f8d"}
WALL = "#4f6272"
GOAL = "#e0a100"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _rooms(ax, frame: bool = False):
    """The two rooms in library units, y down."""
    if frame:
        ax.imshow(W.BACKGROUND.permute(1, 2, 0).numpy(), extent=(0, W.IMG, W.IMG, 0))
    else:
        ax.add_patch(plt.Rectangle((W.BORDER, W.BORDER), W.IMG - 2 * W.BORDER, W.IMG - 2 * W.BORDER,
                                   fill=False, ec=WALL, lw=1.5))
        ax.plot([W.WALL_X] * 2, [W.BORDER, W.DOOR_Y - W.DOOR_HALF], color=WALL, lw=3)
        ax.plot([W.WALL_X] * 2, [W.DOOR_Y + W.DOOR_HALF, W.IMG - W.BORDER], color=WALL, lw=3)
    ax.set_xlim(0, W.IMG)
    ax.set_ylim(W.IMG, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _runs() -> dict:
    out = {}
    for f in sorted((common.RESULTS / "runs").glob("*_s*.json")):
        r = json.loads(f.read_text())
        out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


def world() -> None:
    R = common.read_results("world")
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.5))
    for i, (p, t) in enumerate((([60.0, 150.0], "agent in the left room"), ([112.0, 49.0], "agent in the door"))):
        ax[i].imshow(W.render(np.array([p]))[0].permute(1, 2, 0).numpy())
        ax[i].set_title(f"a 64 x 64 frame: {t}", fontsize=9)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    h = np.array(R["data"]["position_hist_32"], float)
    _rooms(ax[2])
    ax[2].imshow(np.log1p(h), extent=(0, W.IMG, W.IMG, 0), cmap="Blues", alpha=0.9)
    ax[2].set_title(f"where the agent is in the {R['data']['frames']:,} training frames\n"
                    f"(log count; {R['data']['crossing_share']:.0%} of episodes cross the wall)", fontsize=9)
    _rooms(ax[3])
    for path in R["examples"]["episode_paths"]:
        p = np.array(path)
        ax[3].plot(p[:, 0], p[:, 1], lw=1, alpha=0.8)
    e = np.array(R["examples"]["expert_cross_path"])
    ax[3].plot(e[:, 0], e[:, 1], color="k", lw=2, ls="--", label="expert, one test task")
    t = R["examples"]["task_cross"]
    ax[3].plot(*t["start"], "o", color="k")
    ax[3].plot(*t["goal"], "*", color=GOAL, ms=14, mec="k")
    ax[3].legend(fontsize=8, loc="lower right")
    ax[3].set_title("four training episodes (colours) and the expert", fontsize=9)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "world.png", dpi=150)
    plt.close(fig)


def train() -> None:
    S = common.read_results("train")
    runs = _runs()
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for arm, rs in runs.items():
        log = rs[0]["progress"]["log"]
        ax[0].plot([p["step"] for p in log], [p["pred"] for p in log], color=ARM_COL[arm], label=ARM_LABEL[arm])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("training step")
    ax[0].set_title("prediction loss, training batches (seed 0)", fontsize=9)
    ax[0].legend(fontsize=8)
    _style(ax[0])
    arms = [a for a in ARM_LABEL if a in S["arms"]]
    x = np.arange(len(arms))
    for i, a in enumerate(arms):
        ax[1].scatter([i] * len(S["arms"][a]["probe_r2_xy"]), S["arms"][a]["probe_r2_xy"], color=ARM_COL[a], zorder=3)
        ax[1].scatter([i + 0.15] * len(S["arms"][a]["std_norm"]), S["arms"][a]["std_norm"],
                      color=ARM_COL[a], marker="x", zorder=3)
    rw = S["random_weights"]
    ax[1].axhline(rw["probe_r2_xy"], color=ARM_COL["random_weights"], ls=":", lw=1)
    ax[1].text(-0.3, rw["probe_r2_xy"] + 0.02, "probe R², untrained network", fontsize=7, color="#5d6b67")
    ax[1].set_xticks(x, [ARM_LABEL[a] for a in arms], rotation=20, fontsize=8)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title("dots: probe R² for (x, y); crosses: normalised std\n(module 16's collapse meter; held-out frames, three seeds)", fontsize=9)
    _style(ax[1])
    for a in [a for a in arms if a != "pred_only"]:
        r = runs[a][0]["sqdist_by_geodesic"]
        e = np.array(r["edges"], float)
        mid = (e[:-1] + e[1:]) / 2
        med = np.array([np.nan if v is None else v for v in r["median"]])
        ax[2].plot(mid, med / np.nanmax(med), "o-", color=ARM_COL[a], ms=3, label=ARM_LABEL[a])
    ax[2].set_xlabel("walk distance between two frames (library units)")
    ax[2].set_title("median squared latent distance, scaled to its maximum (seed 0;\nprediction only is left out: its latents all sit at one point)", fontsize=9)
    ax[2].legend(fontsize=8)
    _style(ax[2])
    k = np.arange(1, 7)
    for a in arms:
        err = np.mean(S["arms"][a]["rollout_err_units"], 0)
        ax[3].plot(k, err, "o-", color=ARM_COL[a], ms=3, label=ARM_LABEL[a])
    stay = runs["lewm"][0]["stay_put_err_units"]
    ax[3].plot(k, stay, "--", color="#8a8f8d", label="guess: no move")
    ax[3].set_xlabel("blocks imagined (5 steps each)")
    ax[3].set_ylabel("error of the probed position (units)")
    ax[3].set_title("imagined against real position, 1,000 held-out starts", fontsize=9)
    ax[3].legend(fontsize=8)
    _style(ax[3])
    fig.tight_layout()
    fig.savefig(common.FIGURES / "train.png", dpi=150)
    plt.close(fig)


PLAN_ROWS = [("expert__none", "expert (ceiling)"), ("perfect__cem", "CEM, true world, straight-line cost"),
             ("lewm_s{}__cem", "CEM, LeWM D = 32"), ("lewm_s{}__shoot", "random shooting, LeWM D = 32"),
             ("lewm_s{}__grad", "gradient, LeWM D = 32"), ("dim8_s{}__cem", "CEM, D = 8"),
             ("dim192_s{}__cem", "CEM, D = 192"), ("pred_only_s{}__cem", "CEM, prediction only"),
             ("random_weights__cem", "CEM, untrained network"), ("random__none", "random actions (floor)")]


FOLLOW_ROWS = [("lewm_s0__cem_best", "follow-up: CEM, cheapest plan seen, D = 32"),
               ("lewm_s0__cem_noclip", "follow-up: CEM, refit unclipped, D = 32"),
               ("dim192_s0__cem_best", "follow-up: CEM, cheapest plan seen, D = 192")]


def plan_table(P: dict, followup: bool = False) -> list[dict]:
    """One row per agent: mean success over seeds (or the single run) in each split, and the seed values."""
    rows = []
    src = {**P["rows"], **({k: v for k, v in P["followup_rows"].items() if k != "written"} if followup else {})}
    for key, label in PLAN_ROWS + (FOLLOW_ROWS if followup else []):
        names = [key.format(s) for s in (0, 1, 2)] if "{}" in key else [key]
        rs = [src[n] for n in names if n in src]
        if not rs:
            continue
        rows.append({"key": key, "label": label, "n_seeds": len(rs),
                     **{s: [r[s]["success"] for r in rs] for s in ("cross", "same")}})
    return rows


def plan() -> None:
    P = common.read_results("plan")
    rows = plan_table(P, followup=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    y = np.arange(len(rows))[::-1]
    for off, split, col in ((0.18, "same", "#1d6fb8"), (-0.18, "cross", "#d1495b")):
        m = [np.mean(r[split]) for r in rows]
        ax[0].barh(y + off, m, height=0.34, color=col, label={"same": "goal in the same room",
                                                              "cross": "goal in the other room"}[split],
                   hatch=None)
        for yi, r, v in zip(y, rows, m):
            if r["label"].startswith("follow-up"):
                ax[0].barh(yi + off, v, height=0.34, color="white", edgecolor=col, hatch="///")
        for yi, r in zip(y, rows):
            if r["n_seeds"] > 1:
                ax[0].scatter(r[split], [yi + off] * len(r[split]), color="k", s=6, zorder=3)
    ax[0].set_yticks(y, [r["label"] for r in rows], fontsize=8)
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("share of 100 tasks reached (bars: mean of seeds; dots: seeds;\nhatched: one seed, written after the results)")
    ax[0].legend(fontsize=8, loc="lower right")
    _style(ax[0])
    run = json.loads((common.RESULTS / "plan_runs" / "lewm_s0__cem.json").read_text())
    shoot = json.loads((common.RESULTS / "plan_runs" / "lewm_s0__shoot.json").read_text())
    for a, (name, r, col) in zip(ax[1:], (("CEM", run, "#1d6fb8"), ("random shooting", shoot, "#3fa34d"))):
        _rooms(a)
        ep = r["same"]["episodes"][0]
        t = P["tasks"]["same"]["first"][0]
        p = np.array(ep["path"])
        a.plot(p[:, 0], p[:, 1], color=col, lw=2, label="real path")
        for im in ep["imagined"][:6]:
            im = np.array(im)
            a.plot(im[:, 0], im[:, 1], ":", color=col, lw=1)
        a.plot(*t["start"], "o", color="k")
        a.plot(*t["goal"], "*", color=GOAL, ms=14, mec="k")
        a.set_title(f"{name}, D = 32 seed 0, same-room task 1:\n"
                    f"{'reached' if ep['success'] else 'missed'} in {ep['steps']} steps "
                    "(dotted: first six plans, read by the probe)", fontsize=8)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "plan.png", dpi=150)
    plt.close(fig)


GRID_LABEL = {"ema": "module 25 (EMA)", "ema_plain": "EMA, plain predictor", "ema_idm": "EMA + inverse dynamics",
              "vc": "var + cov", "vc_idm": "var + cov + inv. dyn.", "pldm": "PLDM: var + cov + inv. dyn. + smooth",
              "sigreg": "SIGReg", "sigreg_idm": "SIGReg + inv. dyn."}


def grid() -> None:
    G = common.read_results("grid")
    arms = [a for a in GRID_LABEL if a in G["arms"]]
    F = common.read_results("grid_followup")["arms"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.6))
    y = np.arange(len(arms))[::-1]
    for a_, key, title in ((ax[0], "plan_success", "random-shooting planner: goals reached of 300"),
                           (ax[1], "agent_r2", "ridge probe R² for the agent's cell (held out)")):
        m = [np.mean(G["arms"][a][key]) for a in arms]
        a_.barh(y, m, color=["#d1495b" if a.startswith("ema") else "#1d6fb8" if a.startswith(("vc", "pldm")) else "#3fa34d"
                             for a in arms], height=0.6)
        for yi, a in zip(y, arms):
            a_.scatter(G["arms"][a][key], [yi] * len(G["arms"][a][key]), color="k", s=8, zorder=3)
        a_.set_yticks(y, [GRID_LABEL[a] for a in arms], fontsize=8)
        a_.set_title(title, fontsize=9)
        _style(a_)
    ax[0].axvline(G["baselines"]["random"]["success"], color="#8a8f8d", ls="--", lw=1)
    ax[0].text(G["baselines"]["random"]["success"], len(arms) - 0.4, " random actions", fontsize=7, color="#5d6b67")
    ax[0].axvline(0.757, color="#1c2422", ls=":", lw=1)
    ax[0].text(0.757, len(arms) - 0.4, " module 25's one-hot model", fontsize=7)
    ax[0].axvline(G["baselines"]["perfect"]["success"], color="#3fa34d", ls=":", lw=1)
    ax[0].text(G["baselines"]["perfect"]["success"], -0.9, " true world\n as the model", fontsize=7, color="#3fa34d")
    ax[0].set_xlim(0, 1)
    ax[1].set_xlim(-0.05, 1.05)
    fa = [a for a in arms if a in F]
    yf = np.array([y[arms.index(a)] for a in fa])
    ax[2].barh(yf + 0.18, [np.mean(F[a]["agent_probe_plan_success"]) for a in fa], height=0.34, color="#1c2422",
               label="planner reads the agent; the goal cell is given")
    ax[2].barh(yf - 0.18, [np.mean(G["arms"][a]["plan_success"]) for a in fa], height=0.34, color="#bfc5c3",
               label="planner reads the goal offset (as above)")
    for a, yi in zip(fa, yf):
        ax[2].scatter(F[a]["agent_probe_plan_success"], [yi + 0.18] * 3, color="#e0a100", s=8, zorder=3)
    ax[2].set_yticks(y, [GRID_LABEL[a] for a in arms], fontsize=8)
    ax[2].set_xlim(0, 1)
    ax[2].legend(fontsize=7, loc="upper right")
    ax[2].set_title("follow-up, written after the results:\ngoals reached when the goal comes from outside the latent", fontsize=9)
    _style(ax[2])
    fig.tight_layout()
    fig.savefig(common.FIGURES / "grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    world()
    train()
    plan()
    grid()
