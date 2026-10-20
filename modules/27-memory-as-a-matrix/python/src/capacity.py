"""Section 1: how many (key, value) pairs one matrix state holds, with no training.

Write n pairs of random unit vectors (d_k = d_v = 32) into one state by each
rule, then read every key back. Twice: with independent random keys, and with
keys that share a common direction (cosine near 0.5 between any two). A read counts as right when the returned
vector lies closer (by cosine) to its own value than to any other of the n
values. Averaged over 20 draws.

Also: the overwrite test (one key written twice), recall by the age of a pair,
and a check that the delta rule's write is one gradient step.

    uv run python src/capacity.py --figures
"""

from __future__ import annotations

import argparse

import common
import torch

import memory

D = 32
NS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
DRAWS = 20
ALPHA = 0.95
SETTINGS = {
    # label: (rule, alpha, passes)
    "linear": ("linear", 1.0, 1),
    "decay, a = 0.95": ("decay", ALPHA, 1),
    "delta": ("delta", 1.0, 1),
    "gated delta, a = 0.95": ("gated_delta", ALPHA, 1),
    "delta, 10 passes": ("delta", 1.0, 10),
}


RHO = 0.5   # correlated keys: each key shares this much of its square length with one common direction


def draw(n: int, gen: torch.Generator, d: int = D, rho: float = 0.0):
    """n unit keys and n unit values. With rho > 0, keys lean on one shared direction c:
    k = sqrt(rho) c + sqrt(1 - rho) z, normalised; two such keys have cosine near rho."""
    z = memory.unit(torch.randn(n, d, generator=gen, dtype=torch.float64))
    c = memory.unit(torch.randn(1, d, generator=gen, dtype=torch.float64))
    k = memory.unit(rho**0.5 * c + (1 - rho) ** 0.5 * z)
    v = memory.unit(torch.randn(n, d, generator=gen, dtype=torch.float64))
    return k, v


def right(out, values):
    """Per pair: is the read closest to its own value among the n values?"""
    sims = memory.unit(out) @ values.T
    return (sims.argmax(1) == torch.arange(len(values))).double()


def capacity_curves(rho: float = 0.0) -> dict:
    gen = torch.Generator().manual_seed(0)
    curves = {lab: {"recall": [], "cos": []} for lab in SETTINGS}
    for n in NS:
        acc = {lab: [] for lab in SETTINGS}
        cos = {lab: [] for lab in SETTINGS}
        for _ in range(DRAWS):
            k, v = draw(n, gen, rho=rho)
            for lab, (rule, a, passes) in SETTINGS.items():
                out = memory.write_read(k, v, rule, alpha=a, passes=passes)
                acc[lab].append(right(out, v).mean().item())
                cos[lab].append(torch.nn.functional.cosine_similarity(out, v, dim=1).mean().item())
        for lab in SETTINGS:
            curves[lab]["recall"].append(round(sum(acc[lab]) / DRAWS, 4))
            curves[lab]["cos"].append(round(sum(cos[lab]) / DRAWS, 4))
    return curves


def by_age(n: int = 64) -> dict:
    """Recall of each pair by its position in the write order, n = 2 d_k pairs."""
    gen = torch.Generator().manual_seed(1)
    tot = {lab: torch.zeros(n, dtype=torch.float64) for lab in SETTINGS}
    for _ in range(200):
        k, v = draw(n, gen)
        for lab, (rule, a, passes) in SETTINGS.items():
            tot[lab] += right(memory.write_read(k, v, rule, alpha=a, passes=passes), v)
    return {lab: [round(x, 3) for x in (t / 200).tolist()] for lab, t in tot.items()}


def overwrite(n_between: int = 8) -> dict:
    """Write (k1, v_old), n_between other pairs, then (k1, v_new). What does k1 return?"""
    gen = torch.Generator().manual_seed(2)
    rows = {lab: {"cos_new": 0.0, "cos_old": 0.0} for lab in ("linear", "delta", "gated delta, a = 0.95")}
    reps = 200
    for _ in range(reps):
        k, v = draw(n_between + 2, gen)
        keys = torch.cat([k[:1], k[1:-1], k[:1]])                     # k1 again at the end
        vals = torch.cat([v[:1], v[1:-1], v[-1:]])                    # v_old first, v_new last
        for lab in rows:
            rule, a, _ = SETTINGS[lab]
            S_out = memory.write_read(keys, vals, rule, alpha=a)[-1]     # read k1
            rows[lab]["cos_new"] += torch.cosine_similarity(S_out, v[-1], dim=0).item() / reps
            rows[lab]["cos_old"] += torch.cosine_similarity(S_out, v[0], dim=0).item() / reps
    return {lab: {k: round(x, 3) for k, x in r.items()} for lab, r in rows.items()} | {"n_between": n_between}


def delta_is_gradient_step() -> float:
    """Max |delta write - one SGD step on 1/2 ||S k - v||^2| over 100 random cases."""
    gen = torch.Generator().manual_seed(3)
    worst = 0.0
    for _ in range(100):
        S = torch.randn(D, D, generator=gen, dtype=torch.float64, requires_grad=True)
        k, v = draw(1, gen)
        k, v = k[0], v[0]
        beta = torch.rand((), generator=gen, dtype=torch.float64).item()
        loss = 0.5 * ((S @ k - v) ** 2).sum()
        (g,) = torch.autograd.grad(loss, S)
        sgd = S.detach() - beta * g
        rule = S.detach() + beta * torch.outer(v - S.detach() @ k, k)
        worst = max(worst, (sgd - rule).abs().max().item())
    return worst


def first_below(curve, xs, level=0.5):
    for x, y in zip(xs, curve):
        if y < level:
            return x
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = common.Guard("capacity")
    curves = capacity_curves()
    corr = capacity_curves(RHO)
    guard.tick("curves")
    age = by_age()
    over = overwrite()
    gstep = delta_is_gradient_step()
    payload = {
        "d_k": D, "d_v": D, "draws": DRAWS, "alpha": ALPHA, "ns": NS,
        "curves": curves,
        "first_n_below_half": {lab: first_below(c["recall"], NS) for lab, c in curves.items()},
        "rho": RHO, "curves_correlated": corr,
        "first_n_below_half_correlated": {lab: first_below(c["recall"], NS) for lab, c in corr.items()},
        "by_age_n": 64, "by_age": age,
        "overwrite": over,
        "delta_vs_sgd_max_abs_diff": gstep,
        "guard": guard.report(),
    }
    for lab, c in curves.items():
        print(f"{lab:24s} recall " + " ".join(f"{x:.2f}" for x in c["recall"]))
    for lab, c in corr.items():
        print(f"{lab:24s} corr.  " + " ".join(f"{x:.2f}" for x in c["recall"]))
    print("overwrite:", over)
    print(f"delta write vs SGD step: {gstep:.1e}")
    common.write_results("capacity", payload)
    if args.figures:
        import figures

        figures.fig_capacity()


if __name__ == "__main__":
    main()
