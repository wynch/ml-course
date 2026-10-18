"""Section 4: PPO on the hand-written CartPole, 3 seeds.

Proximal policy optimisation (Schulman et al., 2017) with the clipped
objective. Per update: 4 copies of the environment run 128 steps each with
the current policy (512 transitions); advantages come from GAE (lambda 0.95)
on a learned value; then 4 epochs of 4 minibatches maximise

    L = mean( min( rho A, clip(rho, 1 - eps, 1 + eps) A ) ),   rho = pi_new(a|s) / pi_old(a|s)

minus 0.5 x value loss, plus 0.01 x entropy. The clip removes any gain from
moving rho beyond 1 +/- eps. Hyperparameters follow CleanRL's ppo.py
defaults for CartPole (Huang et al., 2022), except the budget.

"Clip fraction" is the share of a minibatch's samples whose ratio lay
outside [1 - eps, 1 + eps] (CleanRL's definition). For those that moved in
the direction their advantage favours, the clip removes the gradient; for
those that moved the other way, the gradient stays.

    uv run python src/ppo.py [--figures]                       # CleanRL's rate, 2.5e-4: results/ppo.json
    uv run python src/ppo.py --lr 2.5e-3 --name ppo_fast       # ten times the rate
    uv run python src/ppo.py --lr 2.5e-3 --no-clip --name ppo_fast_noclip
"""

from __future__ import annotations

import argparse

import numpy as np

from common import Guard, curve_on_grid, write_results
import torch  # noqa: E402  (after common: the MPS guard)

import cartpole as cp

SEEDS = (0, 1, 2)
BUDGET = 200_000
N_ENVS, N_STEPS = 4, 128
GAMMA, LAM = 0.99, 0.95
CLIP = 0.2
EPOCHS, MINIBATCHES = 4, 4
LR = 2.5e-4                     # annealed linearly to 0
ENT_COEF, VF_COEF, MAX_GRAD_NORM = 0.01, 0.5, 0.5
GRID = list(range(5_000, BUDGET + 1, 5_000))
EVAL_EPISODES = 20
RATIO_BINS = np.linspace(0.5, 1.5, 41)


def layer(i, o, std=np.sqrt(2)):
    lin = torch.nn.Linear(i, o)
    torch.nn.init.orthogonal_(lin.weight, std)
    torch.nn.init.zeros_(lin.bias)
    return lin


class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        T = torch.nn.Tanh
        self.critic = torch.nn.Sequential(layer(4, 64), T(), layer(64, 64), T(), layer(64, 1, 1.0))
        self.actor = torch.nn.Sequential(layer(4, 64), T(), layer(64, 64), T(), layer(64, 2, 0.01))

    def value(self, s):
        return self.critic(s).squeeze(-1)

    def dist(self, s):
        return torch.distributions.Categorical(logits=self.actor(s))


def clipped_objective(logp_new, logp_old, adv, clip=CLIP, use_clip=True):
    """PPO's clipped surrogate (to maximise) and the clip fraction.

    With ``use_clip=False`` the objective is the plain ratio-weighted advantage,
    and the clip fraction still counts samples outside [1 - clip, 1 + clip].
    """
    ratio = torch.exp(logp_new - logp_old)
    if use_clip:
        obj = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
    else:
        obj = (ratio * adv).mean()
    clip_frac = ((ratio - 1).abs() > clip).float().mean()
    return obj, clip_frac, ratio


def gae(rewards, values, next_value, dones, gamma=GAMMA, lam=LAM):
    """Generalised advantage estimates for arrays of shape (steps, envs).

    dones[t] is 1 when the episode ended at step t by failure (no value after it);
    a 500-step cut is handled by the caller, which folds the value of the cut
    state into rewards[t].
    """
    T = len(rewards)
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in range(T - 1, -1, -1):
        nv = next_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv


def greedy(agent):
    def act(s):
        with torch.no_grad():
            return int(agent.actor(torch.as_tensor(s, dtype=torch.float32)).argmax())
    return act


def train(seed: int, lr: float = LR, use_clip: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = Agent()
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    envs = cp.VecCartPole(N_ENVS, rng)
    n_updates = BUDGET // (N_ENVS * N_STEPS)
    batch = N_ENVS * N_STEPS
    mb = batch // MINIBATCHES
    ep_ret = np.zeros(N_ENVS)
    ends, rets, log = [], [], []
    total = 0
    s = envs.state.copy()
    for u in range(n_updates):
        opt.param_groups[0]["lr"] = lr * (1 - u / n_updates)
        S = np.zeros((N_STEPS, N_ENVS, 4), np.float32)
        A = np.zeros((N_STEPS, N_ENVS), np.int64)
        LP = np.zeros((N_STEPS, N_ENVS), np.float32)
        V = np.zeros((N_STEPS, N_ENVS), np.float32)
        R = np.zeros((N_STEPS, N_ENVS), np.float32)
        D = np.zeros((N_STEPS, N_ENVS), np.float32)
        for t in range(N_STEPS):
            st = torch.as_tensor(s, dtype=torch.float32)
            with torch.no_grad():
                d = agent.dist(st)
                a = d.sample()
                S[t], A[t], LP[t], V[t] = s, a.numpy(), d.log_prob(a).numpy(), agent.value(st).numpy()
            s, r, term, trunc, final = envs.step(A[t])
            total += N_ENVS
            R[t] = r
            if trunc.any():                         # cut at 500 steps, not failed: bootstrap from the cut state
                with torch.no_grad():
                    R[t, trunc] += GAMMA * agent.value(torch.as_tensor(final[trunc], dtype=torch.float32)).numpy()
            D[t] = term | trunc
            ep_ret += 1
            for i in np.flatnonzero(term | trunc):
                ends.append(total)
                rets.append(float(ep_ret[i]))
                ep_ret[i] = 0
        with torch.no_grad():
            nv = agent.value(torch.as_tensor(s, dtype=torch.float32)).numpy()
        adv = gae(R, V, nv, D)
        ret = adv + V
        bS, bA, bLP = (torch.as_tensor(x.reshape(batch, *x.shape[2:])) for x in (S, A, LP))
        bAdv, bRet = torch.as_tensor(adv.reshape(-1)), torch.as_tensor(ret.reshape(-1))
        fracs, kls, hist = [], [], np.zeros(len(RATIO_BINS) - 1)
        for epoch in range(EPOCHS):
            perm = torch.as_tensor(rng.permutation(batch))
            for k in range(MINIBATCHES):
                idx = perm[k * mb:(k + 1) * mb]
                d = agent.dist(bS[idx])
                logp = d.log_prob(bA[idx])
                a_mb = bAdv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                obj, frac, ratio = clipped_objective(logp, bLP[idx], a_mb, use_clip=use_clip)
                v = agent.value(bS[idx])
                v_loss = 0.5 * ((v - bRet[idx]) ** 2).mean()
                loss = -obj + VF_COEF * v_loss - ENT_COEF * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                opt.step()
                with torch.no_grad():
                    lr_ = logp - bLP[idx]
                    kls.append(float(((torch.exp(lr_) - 1) - lr_).mean()))
                fracs.append(float(frac))
                if epoch == EPOCHS - 1:
                    hist += np.histogram(np.clip(ratio.detach().numpy(), RATIO_BINS[0], RATIO_BINS[-1] - 1e-9),
                                         RATIO_BINS)[0]
        log.append({"update": u + 1, "step": total, "clip_frac": round(float(np.mean(fracs)), 4),
                    "approx_kl": round(float(np.mean(kls)), 5), "ratio_hist_last_epoch": hist.astype(int).tolist()})
    return agent, ends, rets, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--name", default="ppo")
    args = ap.parse_args()
    torch.set_num_threads(1)
    guard = Guard(f"ppo.py --lr {args.lr:g}" + (" --no-clip" if args.no_clip else ""))
    runs = []
    untrained = None
    for seed in SEEDS:
        if seed == 0:
            torch.manual_seed(0)
            fresh = Agent()
            _, untrained = cp.run_episode(greedy(fresh), np.random.default_rng(20_000), record=True)
        agent, ends, rets, log = train(seed, args.lr, not args.no_clip)
        rng = np.random.default_rng(10_000 + seed)
        ev = [cp.run_episode(greedy(agent), rng)[0] for _ in range(EVAL_EPISODES)]
        first500 = next((e for e, r in zip(ends, rets) if r >= cp.MAX_STEPS), None)
        _, states = cp.run_episode(greedy(agent), np.random.default_rng(20_000), record=True)
        runs.append({"seed": seed, "episodes": len(rets), "ends": ends, "returns": rets,
                     "curve": curve_on_grid(ends, rets, GRID), "eval_greedy": ev,
                     "eval_greedy_mean": float(np.mean(ev)), "first_500_at_step": first500,
                     "log": log, "replay_len": len(states) - 1,
                     "replay": np.round(states, 4).tolist() if seed == 0 else None})
        guard.tick(f"seed {seed}")
        fr = [x["clip_frac"] for x in log]
        print(f"seed {seed}: {len(rets)} episodes, last-20 mean {np.mean(rets[-20:]):.1f}, first 500 at step {first500}, "
              f"greedy eval mean {np.mean(ev):.1f}; clip fraction first/max/last {fr[0]:.3f}/{max(fr):.3f}/{fr[-1]:.3f}")
    payload = {"seeds": list(SEEDS), "budget_steps": BUDGET, "n_envs": N_ENVS, "n_steps": N_STEPS,
               "updates": BUDGET // (N_ENVS * N_STEPS), "gamma": GAMMA, "gae_lambda": LAM, "clip": CLIP,
               "epochs": EPOCHS, "minibatches": MINIBATCHES, "lr": args.lr, "use_clip": not args.no_clip, "lr_anneal": "linear to 0",
               "ent_coef": ENT_COEF, "vf_coef": VF_COEF, "max_grad_norm": MAX_GRAD_NORM,
               "grid": GRID, "curve_window": 20, "eval_episodes": EVAL_EPISODES,
               "ratio_bins": RATIO_BINS.round(3).tolist(),
               "untrained_replay": np.round(untrained, 4).tolist(), "untrained_len": len(untrained) - 1,
               "runs": runs, "guard": guard.report()}
    write_results(args.name, payload)
    if args.figures:
        import figures
        figures.main()


if __name__ == "__main__":
    main()
