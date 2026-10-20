"""Section 5: an attribution graph built by hand over the true neurons, then broken by ablation.

The model's neurons are its own units, so the graph needs no replacement model
and has no error nodes. For one prompt and one target (the model's top next
token at the last position):

1. Freeze what is not linear: attention patterns, LayerNorm scales, and which
   8 neurons each TopK keeps. With those frozen, the target logit is linear in
   every neuron's activation, as long as no kept neuron's ReLU input crosses zero.
2. Attribution of an active neuron = activation x gradient of the target logit
   (through every later path). Edges between top neurons: activation x d(later
   neuron)/d(earlier neuron) x the later neuron's gradient.
3. Break it: zero one active neuron at a time and rerun the real model with
   nothing frozen. Compare with the attribution. Then zero the top 10 against
   10 random active neurons, and keep only the top N.

The same runs on the random-weight copy, with its own top next token.

    uv run python src/attribution.py            # results/attribution.json
    uv run python src/attribution.py --figures
"""

from __future__ import annotations

import argparse

from common import Guard, write_results

import numpy as np
import torch

import data
import scores
from analyse_dicts import N_EVAL, true_codes, to_csc
from dicts import get_model

N_PROMPTS, PROMPT_LEN = 20, 40
N_RANDOM_SETS = 100
TOP_SET = 10
KEEP_N = (5, 10, 20, 40, 80)
GRAPH_TOP = 6                        # nodes per layer drawn in the graph
SEED = 0


def prompts(va: np.ndarray, sep: int) -> list[np.ndarray]:
    """Story openings from the held-out tokens after the scoring range: the first 40 tokens after a separator."""
    starts = np.flatnonzero(va[N_EVAL:] == sep) + N_EVAL + 1
    out = []
    for s in starts:
        p = va[s : s + PROMPT_LEN]
        if len(p) == PROMPT_LEN and sep not in p:
            out.append(p)
        if len(out) == N_PROMPTS:
            break
    return out


class Frozen:
    """Context manager: attention patterns, LayerNorm scales and TopK choices fixed at their values on the prompt."""

    def __init__(self, model, x):
        self.model, self.hooks, self.std = model, [], {}
        with torch.no_grad():
            lns = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
            hs = [ln.register_forward_hook(self._grab(ln)) for ln in lns]
            model(x)
            for h in hs:
                h.remove()
            self.patterns = [b.attn.last.detach() for b in model.blocks]
            self.gates = [(b.mlp.h > 0).float() for b in model.blocks]
        self.lns = lns

    def _grab(self, ln):
        def f(m, i, o):
            self.std[ln] = torch.sqrt(i[0].var(-1, keepdim=True, unbiased=False) + ln.eps).detach()
        return f

    def __enter__(self):
        for b, p, g in zip(self.model.blocks, self.patterns, self.gates):
            b.attn.frozen = p
            b.mlp.gate = g
        for ln in self.lns:
            std = self.std[ln]
            self.hooks.append(ln.register_forward_hook(
                lambda m, i, o, std=std: (i[0] - i[0].mean(-1, keepdim=True)) / std * m.weight + m.bias))
        return self

    def __exit__(self, *a):
        for b in self.model.blocks:
            b.attn.frozen = None
            b.mlp.gate = None
        for h in self.hooks:
            h.remove()
        self.hooks = []


def attribute(model, x, target: int):
    """Activation, gradient and attribution of every neuron, both layers: arrays (layers, T, N)."""
    keep = {}

    def leaf(h):
        h = h.detach().requires_grad_(True)
        keep[0] = h
        return h

    def track(h):
        h.retain_grad()
        keep[1] = h
        return h

    fr = Frozen(model, x)
    model.blocks[0].mlp.edit, model.blocks[1].mlp.edit = leaf, track
    try:
        with fr:
            logit = model(x)[0, -1, target]
            logit.backward()
    finally:
        model.blocks[0].mlp.edit = model.blocks[1].mlp.edit = None
    acts = torch.stack([keep[0][0].detach(), keep[1][0].detach()])
    grads = torch.stack([keep[0].grad[0], keep[1].grad[0]])
    return acts.numpy(), grads.numpy(), (acts * grads).numpy(), float(logit.detach())


def edges(model, x, target: int, src: list, dst: list) -> list:
    """Frozen-model edge weight a_src * d h_dst / d h_src * g_dst for layer-0 sources and layer-1 targets."""
    out = []
    acts, grads, _, _ = attribute(model, x, target)
    for (l1, p1, n1) in dst:
        keep = {}

        def leaf(h):
            h = h.detach().requires_grad_(True)
            keep[0] = h
            return h

        def track(h):
            keep[1] = h
            return h

        fr = Frozen(model, x)
        model.blocks[0].mlp.edit, model.blocks[1].mlp.edit = leaf, track
        try:
            with fr:
                model(x)
                keep[1][0, p1, n1].backward()
        finally:
            model.blocks[0].mlp.edit = model.blocks[1].mlp.edit = None
        d = keep[0].grad[0].numpy()
        for (l0, p0, n0) in src:
            w = acts[0, p0, n0] * d[p0, n0] * grads[1, p1, n1]
            out.append({"src": [l0, p0, n0], "dst": [l1, p1, n1], "w": round(float(w), 4)})
    return out


@torch.no_grad()
def run_masked(model, x, zeroed: list, target: int) -> np.ndarray:
    """Target logit at the last position with each set of (layer, pos, neuron) nodes zeroed, nothing frozen."""
    out = []
    L, T, N = len(model.blocks), x.shape[1], model.c.n_mlp
    for s in range(0, len(zeroed), 64):
        m = torch.ones(len(zeroed[s : s + 64]), L, T, N)
        for j, nodes in enumerate(zeroed[s : s + 64]):
            for (l, p, n) in nodes:
                m[j, l, p, n] = 0
        xb = x.expand(len(m), -1)
        model.blocks[0].mlp.edit = lambda h, m=m: h * m[:, 0]
        model.blocks[1].mlp.edit = lambda h, m=m: h * m[:, 1]
        try:
            lg = model(xb)[:, -1]
        finally:
            model.blocks[0].mlp.edit = model.blocks[1].mlp.edit = None
        out.append(lg[:, target].numpy())
    return np.concatenate(out)


@torch.no_grad()
def prob_masked(model, x, mask: torch.Tensor, target: int) -> float:
    model.blocks[0].mlp.edit = lambda h: h * mask[0]
    model.blocks[1].mlp.edit = lambda h: h * mask[1]
    try:
        p = model(x)[0, -1].softmax(-1)[target]
    finally:
        model.blocks[0].mlp.edit = model.blocks[1].mlp.edit = None
    return float(p)


def frozen_check(model, ids: np.ndarray, target: int, n: int = 5) -> float:
    """Largest gap between attribution and the drop when one top neuron is zeroed in the frozen model (should be ~0)."""
    x = torch.from_numpy(ids)[None]
    _, _, attr, logit = attribute(model, x, target)
    gap = 0.0
    for j in np.argsort(-np.abs(attr).ravel())[:n]:
        l, p, k = np.unravel_index(j, attr.shape)
        fr = Frozen(model, x)

        def zero(h, p=p, k=k):
            h = h.clone()
            h[0, p, k] = 0
            return h

        model.blocks[l].mlp.edit = zero
        try:
            with fr, torch.no_grad():
                drop = logit - float(model(x)[0, -1, target])
        finally:
            model.blocks[l].mlp.edit = None
        gap = max(gap, abs(drop - float(attr[l, p, k])))
    return gap


def one_prompt(model, ids: np.ndarray, rng) -> dict:
    x = torch.from_numpy(ids)[None]
    with torch.no_grad():
        lg = model(x)[0, -1]
    target = int(lg.argmax())
    p_full = float(lg.softmax(-1)[target])
    acts, grads, attr, logit = attribute(model, x, target)
    active = np.argwhere(acts > 0)                       # (n_active, 3): layer, position, neuron
    n_act = len(active)
    L, T, N = acts.shape
    eff = logit - run_masked(model, x, [[tuple(a)] for a in active], target)    # drop in the target logit
    att = attr[tuple(active.T)]
    r = float(np.corrcoef(att, eff)[0, 1])
    order = np.argsort(-np.abs(att))
    top = active[order[:TOP_SET]]
    drop_top = float(logit - run_masked(model, x, [[tuple(a) for a in top]], target)[0])
    rand_sets = [[tuple(a) for a in active[rng.choice(n_act, TOP_SET, replace=False)]] for _ in range(N_RANDOM_SETS)]
    drops_rand = logit - run_masked(model, x, rand_sets, target)
    faith = {}
    for k in KEEP_N:
        m = torch.ones(L, T, N)
        for (l, p, n) in active[order[k:]]:
            m[l, p, n] = 0
        faith[str(k)] = round(prob_masked(model, x, m, target) / p_full, 4)
    m0 = torch.ones(L, T, N)
    for (l, p, n) in active:
        m0[l, p, n] = 0
    faith["0"] = round(prob_masked(model, x, m0, target) / p_full, 4)
    return {
        "target": target, "p_full": round(p_full, 4), "logit": round(logit, 4), "n_active": n_act,
        "pearson_r": round(r, 4),
        "drop_top10": round(drop_top, 4), "drop_random10_mean": round(float(drops_rand.mean()), 4),
        "drop_random10_p95": round(float(np.quantile(drops_rand, 0.95)), 4),
        "faithfulness_keep_top": faith,
        "_active": active, "_att": att, "_eff": eff, "_order": order,
    }


def run(figures: bool = False) -> dict:
    guard = Guard("attribution.py")
    _, va = data.tokens()
    vocab = data.Vocab()
    pieces = vocab.pieces()
    sep = vocab.encode("<|endoftext|>")[0]
    ps = prompts(va, sep)
    out = {"n_prompts": len(ps), "prompt_len": PROMPT_LEN, "top_set": TOP_SET, "random_sets": N_RANDOM_SETS,
           "models": {}}
    for which in ("trained", "random"):
        model = get_model(which, "cpu")
        rng = np.random.default_rng(SEED)
        rows, all_a, all_e = [], [], []
        for i, p in enumerate(ps):
            r = one_prompt(model, p, rng)
            all_a.append(r["_att"]); all_e.append(r["_eff"])
            if i == 0:
                first = r
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        A, E = np.concatenate(all_a), np.concatenate(all_e)
        ratio = [r["drop_top10"] / r["drop_random10_mean"] if r["drop_random10_mean"] > 0 else None for r in rows]
        M = {
            "prompts": rows,
            "pearson_r_pooled": round(float(np.corrcoef(A, E)[0, 1]), 4),
            "pearson_r_median": round(float(np.median([r["pearson_r"] for r in rows])), 4),
            "top10_beats_random_mean": int(sum(r["drop_top10"] > r["drop_random10_mean"] for r in rows)),
            "top10_beats_random_p95": int(sum(r["drop_top10"] > r["drop_random10_p95"] for r in rows)),
            "median_drop_top10": round(float(np.median([r["drop_top10"] for r in rows])), 4),
            "median_drop_random10": round(float(np.median([r["drop_random10_mean"] for r in rows])), 4),
            "faithfulness_median": {k: round(float(np.median([r["faithfulness_keep_top"][k] for r in rows])), 4)
                                    for k in ["0"] + [str(k) for k in KEEP_N]},
            "scatter": {"att": np.round(A, 4).tolist(), "eff": np.round(E, 4).tolist()} if which == "trained" else
                       {"att": np.round(A, 4).tolist(), "eff": np.round(E, 4).tolist()},
        }
        M["frozen_check_max_gap"] = float(f"{frozen_check(model, ps[0], first['target']):.2e}")
        M["median_ratio_top_vs_random"] = round(float(np.median([x for x in ratio if x is not None])), 2)
        # The graph for prompt 0.
        act, att, order = first["_active"], first["_att"], first["_order"]
        top_nodes = {0: [], 1: []}
        for j in order:
            l = int(act[j][0])
            if len(top_nodes[l]) < GRAPH_TOP:
                top_nodes[l].append(j)
        codes = {l: true_codes(model, va, "cpu", layer=l) for l in (0, 1)}
        Tm = {l: to_csc(*codes[l], model.c.n_mlp) for l in (0, 1)}
        nodes = []
        for l in (0, 1):
            for j in top_nodes[l]:
                _, pos, n = (int(v) for v in act[j])
                det, expl = scores.detection(Tm[l][:, n].toarray().ravel(), va[:N_EVAL])
                nodes.append({"layer": l, "pos": pos, "neuron": n, "token": pieces[ps[0][pos]],
                              "att": round(float(att[j]), 4), "eff": round(float(first["_eff"][j]), 4),
                              "expl": [] if expl is None else [pieces[t] for t in expl],
                              "det": None if det is None else round(det, 3)})
        src = [(0, d["pos"], d["neuron"]) for d in nodes if d["layer"] == 0]
        dst = [(1, d["pos"], d["neuron"]) for d in nodes if d["layer"] == 1]
        M["graph"] = {"prompt": [pieces[t] for t in ps[0]], "target": pieces[first["target"]],
                      "p_full": first["p_full"], "logit": first["logit"], "nodes": nodes,
                      "edges": edges(model, torch.from_numpy(ps[0])[None], first["target"], src, dst)}
        for r in rows:
            r["target_piece"] = pieces[r["target"]]
        out["models"][which] = M
        print(f"{which}: pooled r {M['pearson_r_pooled']}, median r {M['pearson_r_median']}, "
              f"top10>random {M['top10_beats_random_mean']}/{len(rows)}, faithfulness {M['faithfulness_median']}")
        guard.tick(which)
    out["prompts_text"] = [[pieces[t] for t in p] for p in ps]
    out["guard"] = guard.report()
    write_results("attribution", out)
    if figures:
        import figures as fg
        fg.fig_attribution(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    run(ap.parse_args().figures)
