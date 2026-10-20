"""Section 2 and 3: RL with a checker, by hand, on the base char-GPT.

Each step draws 32 training problems, samples a group of G = 8 replies to
each at temperature 1, scores every reply with the checker (task.check), turns
the 8 rewards of a group into 8 advantages, and takes one Adam step on a
policy-gradient loss. The arms differ only in that last part:

| arm          | advantage of reply i                  | how a reply's characters are summed |
|--------------|---------------------------------------|-------------------------------------|
| grpo         | (r_i - mean) / (std + 1e-4)           | mean over the reply's own length    |
| drgrpo       | r_i - mean                            | sum, divided by MAX_REPLY for all   |
| rloo         | r_i - mean of the other 7             | sum, divided by MAX_REPLY for all   |
| random       | as grpo, on coin-flip rewards         | as grpo                             |
| random_clip  | as random, 4 clipped updates per batch| as grpo                             |

``random`` flips a coin for each reply with the batch's true success rate,
so it hands out as many 1s as the checker would, at random. ``random_clip``
reuses each batch for 4 Adam steps with PPO's clip (epsilon 0.2): the setting
in which Shao et al. (2025) trace their random-reward gains to the clip.
With one step per batch (every other arm) the ratio is 1 and the clip never
acts.

Each run writes ``results/runs/<arm>_s<seed>.json``: a log every 10 steps,
test accuracy every 50, and at the end 128 samples on each of the 500 test
problems, for pass@k. ``--base`` writes the same final numbers for the base
model. Runs go in chunks: ``--all`` starts a run only if it should end
before 4.75 minutes have gone, then stops; run it again until it prints ``remaining: 0``.
"""

from __future__ import annotations

import argparse
import time

import common  # noqa: F401  (laptop guard before torch)
import numpy as np
import torch

import rollout
import task
from common import RESULTS, Guard, write_results
from sft import load_base

ARMS = ["grpo", "drgrpo", "rloo", "random", "random_clip"]
SEEDS = [0, 1, 2]
STEPS = 300
PROMPTS = 32          # problems per step
G = 8                 # replies per problem
LR = 5e-5
EPS = 0.2             # PPO clip, used only when INNER > 1
N_EVAL = 128          # samples per test problem at the end
KS = [1, 2, 4, 8, 16, 32, 64, 128]
RUNS = RESULTS / "runs"
CHUNK_SECONDS = 285
RUN_SECONDS = {"grpo": 150, "drgrpo": 150, "rloo": 150, "random": 150, "random_clip": 240}


# ------------------------------------------------------------ the advantages
def adv_group(r: torch.Tensor) -> torch.Tensor:
    """GRPO: (r - group mean) / (group std + 1e-4). r is (prompts, G); std as torch's default, n - 1."""
    return (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)


def adv_no_std(r: torch.Tensor) -> torch.Tensor:
    """Dr. GRPO: r - group mean."""
    return r - r.mean(1, keepdim=True)


def adv_loo(r: torch.Tensor) -> torch.Tensor:
    """RLOO: r_i - mean of the other G - 1 rewards of the group."""
    g = r.shape[1]
    return r - (r.sum(1, keepdim=True) - r) / (g - 1)


ADVANTAGE = {"grpo": adv_group, "drgrpo": adv_no_std, "rloo": adv_loo,
             "random": adv_group, "random_clip": adv_group}
LENGTH_NORM = {"grpo": True, "drgrpo": False, "rloo": False, "random": True, "random_clip": True}
INNER = {"grpo": 1, "drgrpo": 1, "rloo": 1, "random": 1, "random_clip": 4}


# ------------------------------------------------------------------ the loss
def pg_loss(lp: torch.Tensor, lp_old: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
            length_norm: bool, eps: float = EPS) -> torch.Tensor:
    """PPO-clip policy-gradient loss over replies, (N, T) per-character tensors, adv (N,).

    length_norm=True: each reply's characters are averaged over its own length, then replies
    averaged (GRPO). False: characters summed and divided by MAX_REPLY for every reply,
    then replies averaged (Dr. GRPO; RLOO's sequence-level sum, scaled the same way).
    """
    ratio = torch.exp(lp - lp_old)
    a = adv[:, None]
    per_tok = -torch.minimum(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a) * mask
    if length_norm:
        per_seq = per_tok.sum(1) / mask.sum(1).clamp(min=1)
    else:
        per_seq = per_tok.sum(1) / task.MAX_REPLY
    return per_seq.mean()


# --------------------------------------------------------------- measuring
@torch.no_grad()
def kl_to_base(model, base, prompts, replies, mask) -> float:
    """Exact KL(policy || base) per reply: summed over characters, over the full 16-letter next-character distribution."""
    _, lp = rollout.token_logprobs(model, prompts, replies, with_logits=True)
    _, lq = rollout.token_logprobs(base, prompts, replies, with_logits=True)
    p = lp.exp()
    kl = (p * (lp - lq)).nan_to_num(0.0).sum(-1)          # PAD: 0 * (-inf - -inf)
    return float((kl * mask).sum(1).mean())


@torch.no_grad()
def final_eval(model, test: np.ndarray, seed: int) -> dict:
    """128 samples per test problem: right counts per problem, pass@k, lengths, style; greedy accuracy."""
    gen = torch.Generator().manual_seed(10_000 + seed)
    P = rollout.prompt_ids(test)
    reps, mask = rollout.sample(model, P.repeat_interleave(N_EVAL, 0), gen=gen)
    r, texts = rollout.rewards(reps, np.repeat(test, N_EVAL, 0))
    right = r.view(len(test), N_EVAL).sum(1).long().tolist()
    lengths = mask.sum(1).numpy()
    worked = np.array([task.is_worked(s) for s in texts])
    rr = r.numpy().astype(bool)
    g, _ = rollout.sample(model, P, greedy=True)
    gr, _ = rollout.rewards(g, test)
    # Distinct final answers per problem: how spread the samples are.
    finals = [s[:-1].rsplit("]", 1)[-1] if s.endswith(".") else None for s in texts]
    distinct = [len({f for f in finals[i * N_EVAL:(i + 1) * N_EVAL]}) for i in range(len(test))]
    return {
        "n_samples": N_EVAL,
        "right_per_problem": right,
        "pass_at_k": {str(k): float(np.mean([task.pass_at_k(N_EVAL, c, k) for c in right])) for k in KS},
        "greedy_acc": float(gr.mean()),
        "mean_length": float(lengths.mean()),
        "mean_length_right": float(lengths[rr].mean()) if rr.any() else None,
        "mean_length_wrong": float(lengths[~rr].mean()) if (~rr).any() else None,
        "worked_share": float(worked.mean()),
        "acc_direct": float(rr[~worked].mean()) if (~worked).any() else None,
        "acc_worked": float(rr[worked].mean()) if worked.any() else None,
        "distinct_answers_mean": float(np.mean(distinct)),
        "examples": texts[:: N_EVAL][:12],
    }


# ------------------------------------------------------------------ one run
def run(arm: str, seed: int) -> dict:
    g = Guard(f"rl_{arm}_s{seed}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(100 + seed)
    gen = torch.Generator().manual_seed(seed)
    coin = torch.Generator().manual_seed(1_000 + seed)
    test, draw = task.splits()
    base = load_base()
    model = load_base()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    log, test_curve = [], []

    for step in range(0, STEPS + 1):
        if step % 50 == 0:
            idx = test[:200]
            reps, _ = rollout.sample(model, rollout.prompt_ids(idx).repeat_interleave(4, 0),
                                     gen=torch.Generator().manual_seed(500 + step))
            rt, _ = rollout.rewards(reps, np.repeat(idx, 4, 0))
            test_curve.append({"step": step, "sample_acc_first_200": float(rt.mean())})
        if step == STEPS:
            break
        pairs = draw(PROMPTS, rng)
        P = rollout.prompt_ids(pairs).repeat_interleave(G, 0)
        reps, mask = rollout.sample(model, P, gen=gen)
        r_true, texts = rollout.rewards(reps, np.repeat(pairs, G, 0))
        if arm.startswith("random"):
            r = (torch.rand(r_true.shape, generator=coin) < r_true.mean()).float()
        else:
            r = r_true
        adv = ADVANTAGE[arm](r.view(PROMPTS, G)).reshape(-1)
        with torch.no_grad():
            lp_old = rollout.token_logprobs(model, P, reps)
        model.train()
        for _ in range(INNER[arm]):
            lp = rollout.token_logprobs(model, P, reps)
            loss = pg_loss(lp, lp_old, adv, mask, LENGTH_NORM[arm])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % 10 == 0:
            L = mask.sum(1).numpy()
            rt = r_true.numpy().astype(bool)
            spread = r.view(PROMPTS, G).std(1)
            log.append({
                "step": step,
                "reward": float(r_true.mean()),
                "reward_used": float(r.mean()),
                "length": float(L.mean()),
                "length_right": float(L[rt].mean()) if rt.any() else None,
                "length_wrong": float(L[~rt].mean()) if (~rt).any() else None,
                "worked_share": float(np.mean([task.is_worked(s) for s in texts])),
                "dead_groups": float((spread == 0).float().mean()),
                "kl_to_base": kl_to_base(model, base, P, reps, mask),
            })
    g.tick("train")
    ev = final_eval(model, test, seed)
    g.tick("evaluate")
    out = {"arm": arm, "seed": seed,
           "config": {"steps": STEPS, "prompts": PROMPTS, "group": G, "lr": LR, "inner": INNER[arm],
                      "eps": EPS, "length_norm": LENGTH_NORM[arm], "temperature": 1.0},
           "log": log, "test_curve": test_curve, "final": ev, "guard": g.report()}
    write_results(f"{arm}_s{seed}", out, RUNS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="evaluate the base model only")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="run missing (arm, seed) pairs for up to 4.5 minutes")
    args = ap.parse_args()
    if args.base:
        g = Guard("rl_base_eval")
        test, _ = task.splits()
        ev = final_eval(load_base(), test, seed=0)
        write_results("base_eval", {"final": ev, "guard": g.report()}, RUNS)
        return
    if args.arm:
        run(args.arm, args.seed)
        return
    if args.all:
        t0 = time.perf_counter()
        todo = [(a, s) for s in SEEDS for a in ARMS if not (RUNS / f"{a}_s{s}.json").exists()]
        # Start a run only if it should end inside the chunk (measured: about 2.3 minutes a run).
        while todo and time.perf_counter() - t0 + RUN_SECONDS[todo[0][0]] < CHUNK_SECONDS:
            run(*todo.pop(0))
        print(f"remaining: {len(todo)}")


if __name__ == "__main__":
    main()
