"""Score the twelve dictionaries and the true neurons; write results/dicts.json.

Sections 2 to 4 of the README read this file:

- ground truth: each dictionary's decoder against block 1's true value vectors (MMCS, share recovered);
- the random-weight copy: the same, plus detection and token R2 for latents and for true neurons;
- seeds: one-to-one matching of seeds 0, 1 and 2, and whether the latents that match the truth are the stable ones.

All scoring runs on the first 2^18 held-out tokens (2,048 windows of 128).

    uv run python src/analyse_dicts.py            # results/dicts.json
    uv run python src/analyse_dicts.py --figures  # and the figures
"""

from __future__ import annotations

import argparse

from common import ARTIFACTS, Guard, get_device, write_results

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import data
import scores
from dicts import KINDS, LAYER, MODELS, N_LAT, SEEDS, TopKDict, get_model, path_for, record
from model import load

N_EVAL = 2**18
ALIVE = 1e-5
MATCH, RECOVER = 0.7, 0.9
N_BROWSE = 40
SEED = 0


def load_dict(model: str, kind: str, seed: int, device) -> TopKDict:
    ck = torch.load(path_for(model, kind, seed), map_location="cpu", weights_only=True)
    st = ck["state"]
    d = TopKDict(st["W_enc"].shape[1], N_LAT, 8, st["b_pre"], st["b_dec"])
    d.load_state_dict(st)
    return d.to(device).eval()


def to_csc(idx: np.ndarray, val: np.ndarray, n_units: int) -> sp.csc_matrix:
    rows = np.repeat(np.arange(len(idx)), idx.shape[1])
    m = sp.csr_matrix((val.ravel(), (rows, idx.ravel())), shape=(len(idx), n_units))
    m.eliminate_zeros()
    return m.tocsc()


@torch.no_grad()
def topk_codes(z: torch.Tensor, k: int = 8):
    v, i = z.topk(k, dim=-1)
    return i.cpu().numpy(), v.cpu().numpy()


@torch.no_grad()
def true_codes(model, ids, device, layer=LAYER):
    """The model's own sparse code at block ``layer``: (idx, val), each (N_EVAL, 8)."""
    ctx = model.c.ctx
    n = N_EVAL // ctx
    x = torch.from_numpy(ids[: n * ctx].reshape(n, ctx))
    I, V = [], []
    for s in range(0, n, 64):
        model(x[s : s + 64].to(device))
        i, v = topk_codes(model.blocks[layer].mlp.h.flatten(0, 1))
        I.append(i); V.append(v)
    return np.concatenate(I), np.concatenate(V)


@torch.no_grad()
def dense_neurons(device, ids):
    m = load(ARTIFACTS / "dense.pt", device)
    ctx = m.c.ctx
    n = N_EVAL // ctx
    x = torch.from_numpy(ids[: n * ctx].reshape(n, ctx))
    out = []
    for s in range(0, n, 256):
        m(x[s : s + 256].to(device))
        out.append(m.blocks[LAYER].mlp.h.flatten(0, 1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def lm_loss(model, ids, device, mlp_out=None) -> float:
    """Held-out next-token loss; ``mlp_out(x_in, y) -> y'`` replaces block 1's MLP output."""
    ctx = model.c.ctx
    n = N_EVAL // ctx
    x = torch.from_numpy(ids[: n * ctx].reshape(n, ctx))
    y = torch.from_numpy(ids[1 : n * ctx + 1].reshape(n, ctx))
    blk = model.blocks[LAYER]
    h = blk.mlp.register_forward_hook(lambda m, i, o: mlp_out(i[0], o)) if mlp_out else None
    tot = 0.0
    try:
        for s in range(0, n, 32):              # logits of 32 windows: 64 MB
            lg = model(x[s : s + 32].to(device))
            tot += float(F.cross_entropy(lg.flatten(0, 1), y[s : s + 32].to(device).flatten(), reduction="sum"))
    finally:
        if h:
            h.remove()
    return tot / (n * ctx)


def contexts(a: np.ndarray, ids: np.ndarray, pieces: list[str], n: int = 5, before: int = 10) -> list:
    """The n strongest firings of a unit on half A, each with ``before`` tokens of context."""
    h = len(a) // 2
    order = np.argsort(-a[:h])[: n * 20]
    out, used = [], []
    for p in order:
        if a[p] <= 0 or len(out) == n:
            break
        if any(abs(p - q) < 8 for q in used):
            continue
        used.append(p)
        start = max(p - before, (p // 128) * 128)        # stay inside the window the model saw
        out.append({"toks": [pieces[t] for t in ids[start : p + 1]], "act": round(float(a[p]), 3)})
    return out


def run(figures: bool = False) -> dict:
    guard = Guard("analyse_dicts.py")
    dev = get_device()
    _, va = data.tokens()
    ids = va[:N_EVAL]
    vocab = data.Vocab()
    pieces = vocab.pieces()
    V = len(pieces)
    out = {"n_eval_tokens": N_EVAL, "alive_rule": f"fires on at least {ALIVE} of the {N_EVAL} held-out tokens",
           "match_threshold": MATCH, "recover_threshold": RECOVER, "models": {}}
    browse = {}
    for which in MODELS:
        model = get_model(which, dev)
        blk = model.blocks[LAYER].mlp
        truth_val = blk.W_out.weight.detach().cpu().numpy()            # (128, 2048): column i is neuron i's value
        truth_key = blk.W_in.weight.detach().cpu().numpy().T           # (128, 2048): column i is neuron i's key
        ti, tv = true_codes(model, ids, dev)
        T = to_csc(ti, tv, N_LAT)
        t_freq = np.asarray((T > 0).sum(0)).ravel() / N_EVAL
        alive = t_freq >= ALIVE
        x, y = record(model, ids, dev, N_EVAL)
        M = {"true_alive": int(alive.sum()), "true_dead": int((t_freq == 0).sum())}
        if which == "trained":
            M["loss"] = {"intact": round(lm_loss(model, va, dev), 4)}
            ymean = y.mean(0)
            M["loss"]["mlp_mean_ablated"] = round(lm_loss(model, va, dev, lambda xi, yo: ymean.expand_as(yo)), 4)
        # True neurons, scored like any unit.
        det, r2, expl = scores.score_units(lambda i: T[:, i].toarray().ravel(), ids, V, N_LAT)
        M["true_neurons"] = {"detection": scores.summary(det), "token_r2": scores.summary(r2)}
        guard.tick(f"{which} truth")
        dicts = {}
        for kind in KINDS:
            K = {}
            decs, alive_of = {}, {}
            for seed in SEEDS:
                d = load_dict(which, kind, seed, dev)
                xin = y if kind == "sae" else x
                with torch.no_grad():               # batch by batch: the dense code would be 2 GB
                    se, I, Vv = 0.0, [], []
                    for s in range(0, len(xin), 16384):
                        z = d.encode(xin[s : s + 16384])
                        se += float(((y[s : s + 16384] - d.decode(z)) ** 2).sum())
                        i_, v_ = topk_codes(z)
                        I.append(i_); Vv.append(v_)
                    fvu = se / float(((y - y.mean(0)) ** 2).sum())
                    zi, zv = np.concatenate(I), np.concatenate(Vv)
                Z = to_csc(zi, zv, N_LAT)
                z_freq = np.asarray((Z > 0).sum(0)).ravel() / N_EVAL
                dec = d.W_dec.detach().cpu().numpy()
                decs[seed] = dec
                alive_of[seed] = z_freq >= ALIVE
                bc = scores.best_cosines(truth_val[:, alive], dec)
                lat_alive = z_freq >= ALIVE
                lat_best = scores.best_cosines(dec[:, lat_alive], truth_val[:, alive])
                row = {"fvu": round(fvu, 4), "dead": int((z_freq == 0).sum()), "alive": int(lat_alive.sum()),
                       "mmcs_value": round(float(bc.mean()), 4),
                       "recovered_share": round(float((bc >= RECOVER).mean()), 4),
                       "latents_matching_truth_share": round(float((lat_best >= RECOVER).mean()), 4),
                       "best_cos_hist": np.histogram(bc, bins=20, range=(0, 1))[0].tolist()}
                if kind == "transcoder":
                    enc = d.W_enc.detach().cpu().numpy().T
                    row["mmcs_key"] = round(float(scores.best_cosines(truth_key[:, alive], enc).mean()), 4)
                if which == "trained" and seed == 0:
                    if kind == "sae":
                        f = lambda xi, yo, d=d: d(yo.flatten(0, 1))[0].view_as(yo)
                    else:
                        f = lambda xi, yo, d=d: d(xi.flatten(0, 1))[0].view_as(yo)
                    row["loss_spliced"] = round(lm_loss(model, va, dev, f), 4)
                if seed == 0:
                    det, r2, expl_l = scores.score_units(lambda i: Z[:, i].toarray().ravel(), ids, V, N_LAT)
                    row["detection"] = scores.summary(det)
                    row["token_r2"] = scores.summary(r2)
                    # Per-latent records for S2 and the explorable.
                    allbest = (scores.unit(dec).T @ scores.unit(truth_val[:, alive]))
                    best_true = np.flatnonzero(alive)[allbest.argmax(1)]
                    s0 = dict(Z=Z, det=det, r2=r2, expl=expl_l, best=allbest.max(1), best_true=best_true,
                              alive=lat_alive)
                K[f"s{seed}"] = row
                print(f"{which} {kind} s{seed}: FVU {row['fvu']}, MMCS {row['mmcs_value']}, "
                      f"recovered {row['recovered_share']}, dead {row['dead']}")
            # Seeds.
            pairs = {}
            for a, b in ((0, 1), (0, 2), (1, 2)):
                mc = scores.seed_match(decs[a], decs[b])
                pairs[f"{a}-{b}"] = round(float((mc[alive_of[a]] >= MATCH).mean()), 4)
            mc01 = scores.seed_match(decs[0], decs[1])
            al = s0["alive"]
            on_truth = al & (s0["best"] >= RECOVER)
            off_truth = al & (s0["best"] < 0.5)
            K["seeds"] = {
                "shared_share_pairs": pairs,
                "shared_share_mean": round(float(np.mean(list(pairs.values()))), 4),
                "rule": f"alive latents of the pair's first seed whose Hungarian partner in the second has cosine >= {MATCH}",
                "matched_cos_hist": np.histogram(mc01[al], bins=20, range=(0, 1))[0].tolist(),
                "on_truth": {"n": int(on_truth.sum()), "shared_share": round(float((mc01[on_truth] >= MATCH).mean()), 4) if on_truth.any() else None},
                "off_truth": {"n": int(off_truth.sum()), "shared_share": round(float((mc01[off_truth] >= MATCH).mean()), 4) if off_truth.any() else None},
                "middle": {"n": int((al & ~on_truth & ~off_truth).sum()),
                           "shared_share": round(float((mc01[al & ~on_truth & ~off_truth] >= MATCH).mean()), 4) if (al & ~on_truth & ~off_truth).any() else None},
            }
            K["mmcs_mean"] = round(float(np.mean([K[f"s{s}"]["mmcs_value"] for s in SEEDS])), 4)
            K["mmcs_spread"] = round(float(np.ptp([K[f"s{s}"]["mmcs_value"] for s in SEEDS])), 4)
            K["recovered_mean"] = round(float(np.mean([K[f"s{s}"]["recovered_share"] for s in SEEDS])), 4)
            dicts[kind] = K
            # Browse records.
            rng = np.random.default_rng(SEED)
            pick = rng.permutation(np.flatnonzero(al & ~np.isnan(s0["det"])))[:N_BROWSE]
            rows = []
            for i in pick:
                a = s0["Z"][:, i].toarray().ravel()
                tn = int(s0["best_true"][i])
                ta = T[:, tn].toarray().ravel()
                tdet, texpl = scores.detection(ta, ids)
                rows.append({
                    "id": int(i), "freq": round(float((a > 0).mean()), 5),
                    "det": round(float(s0["det"][i]), 3), "r2": round(float(s0["r2"][i]), 3),
                    "expl": [pieces[t] for t in s0["expl"][i]],
                    "ctx": contexts(a, ids, pieces),
                    "seed_cos": round(float(mc01[i]), 3),
                    "truth": {"id": tn, "cos": round(float(s0["best"][i]), 3),
                              "det": None if tdet is None else round(tdet, 3),
                              "expl": [] if texpl is None else [pieces[t] for t in texpl],
                              "ctx": contexts(ta, ids, pieces)},
                })
            browse[f"{which}/{kind}"] = rows
            guard.tick(f"{which} {kind}")
        M["dicts"] = dicts
        if which == "trained" and all(path_for("trained", k, "long").exists() for k in KINDS):
            M["long_training"] = {"note": "added after the first results, not pre-registered: seed 0 at 10,000 steps"}
            for kind in KINDS:
                d = load_dict("trained", kind, "long", dev)
                xin = y if kind == "sae" else x
                with torch.no_grad():
                    se, I, Vv = 0.0, [], []
                    for s in range(0, len(xin), 16384):
                        z = d.encode(xin[s : s + 16384])
                        se += float(((y[s : s + 16384] - d.decode(z)) ** 2).sum())
                        i_, v_ = topk_codes(z)
                        I.append(i_); Vv.append(v_)
                Z = to_csc(np.concatenate(I), np.concatenate(Vv), N_LAT)
                z_freq = np.asarray((Z > 0).sum(0)).ravel() / N_EVAL
                bc = scores.best_cosines(truth_val[:, alive], d.W_dec.detach().cpu().numpy())
                M["long_training"][kind] = {
                    "fvu": round(se / float(((y - y.mean(0)) ** 2).sum()), 4), "dead": int((z_freq == 0).sum()),
                    "mmcs_value": round(float(bc.mean()), 4), "recovered_share": round(float((bc >= RECOVER).mean()), 4),
                    "steps": int(torch.load(path_for("trained", kind, "long"), weights_only=True)["steps"])}
                print(f"long {kind}: {M['long_training'][kind]}")
        out["models"][which] = M
        del model, x, y
    # Dense baseline neurons.
    Dn = dense_neurons(dev, ids)
    det, r2, _ = scores.score_units(lambda i: Dn[:, i], ids, V, Dn.shape[1])
    out["dense_neurons"] = {"n": int(Dn.shape[1]), "detection": scores.summary(det), "token_r2": scores.summary(r2)}
    del Dn
    guard.tick("dense")
    out["guard"] = guard.report()
    write_results("dicts", out)
    write_results("browse", {"rule": f"{N_BROWSE} alive, scorable seed-0 latents per dictionary, "
                                      f"numpy.random.default_rng({SEED}).permutation order", "latents": browse})
    if figures:
        import figures as fg
        fg.fig_truth(out); fg.fig_controls(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    run(ap.parse_args().figures)
