"""Section 1b: a linear reward head on frozen SmolLM2-360M-Instruct, layer 18.

Each reply, in the chat template after its prompt, goes through the first 18
of the model's 32 blocks. The residual stream there (960 numbers) at the
reply's last token is the reply's feature vector h. The reward is linear,
r = w · h, and the Bradley-Terry loss on r(chosen) - r(rejected) trains w on
2,000 real pairs. The model never changes; only w (960 numbers) is learned.

Held-out: 500 pairs from the test split, never seen in training. A second
feature, the mean residual over the reply's tokens, is reported beside the
last-token one; the last token was chosen as the main feature before the run.

    uv run python src/reward_head.py --figures

Takes the GPU lock: it loads the model.
"""

from __future__ import annotations

import argparse

from common import FIGURES, LAYER, Guard, free, get_device, load_hf_model, load_tokenizer, write_results

import numpy as np
import torch

import data

BATCH = 16
L2_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
FOLDS = 5


def chat_ids(tok, prompt: str, reply: str) -> tuple[list[int], int]:
    """Token ids of the templated (prompt, reply) and the index where the reply's turn starts."""
    head = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                   add_generation_prompt=True, return_dict=False)
    full = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}],
                                   tokenize=True, return_dict=False)
    assert full[:len(head)] == head
    return full, len(head)


@torch.no_grad()
def residuals(model, tok, texts: list[tuple[str, str]], device, layer: int = LAYER,
              batch: int = BATCH) -> tuple[np.ndarray, np.ndarray]:
    """(last-token, reply-mean) residuals after ``layer`` blocks, each (len(texts), 960).

    Runs only the first ``layer`` blocks: a hook on block layer-1 raises to stop the forward pass.
    """
    enc = [chat_ids(tok, p, r) for p, r in texts]
    order = np.argsort([len(e[0]) for e in enc])
    last = np.zeros((len(texts), model.config.hidden_size), np.float32)
    mean = np.zeros_like(last)
    grabbed = {}

    class Stop(Exception):
        pass

    def hook(_m, _inp, out):
        grabbed["h"] = out[0] if isinstance(out, tuple) else out
        raise Stop

    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        for b0 in range(0, len(order), batch):
            idx = order[b0:b0 + batch]
            L = max(len(enc[i][0]) for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for k, i in enumerate(idx):
                ids[k, :len(enc[i][0])] = torch.tensor(enc[i][0])
                mask[k, :len(enc[i][0])] = 1
            try:
                model.model(input_ids=ids.to(device), attention_mask=mask.to(device))
            except Stop:
                pass
            h = grabbed["h"].float().cpu()
            for k, i in enumerate(idx):
                n, start = len(enc[i][0]), enc[i][1]
                last[i] = h[k, n - 1].numpy()
                mean[i] = h[k, start:n].mean(0).numpy()
    finally:
        handle.remove()
    return last, mean


def fit_head(d: np.ndarray, l2: float, steps: int = 300) -> np.ndarray:
    """w minimising mean(-log sigmoid(d @ w)) + l2/2 |w|^2, by L-BFGS. d = h_chosen - h_rejected, standardised."""
    X = torch.tensor(d, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.softplus(-(X @ w)).mean() + 0.5 * l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()


def pair_acc(d: np.ndarray, w: np.ndarray) -> float:
    return float(((d @ w) > 0).mean())


def cv_pick(d: np.ndarray, seed: int = 0) -> tuple[float, dict]:
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(d)), FOLDS)
    scores = {}
    for l2 in L2_GRID:
        accs = []
        for k in range(FOLDS):
            tr = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
            accs.append(pair_acc(d[folds[k]], fit_head(d[tr], l2)))
        scores[l2] = round(float(np.mean(accs)), 4)
    return max(scores, key=scores.get), scores


def run_feature(name: str, Hc_tr, Hr_tr, Hc_te, Hr_te) -> dict:
    mu = np.concatenate([Hc_tr, Hr_tr]).mean(0)
    sd = np.concatenate([Hc_tr, Hr_tr]).std(0) + 1e-6
    z = lambda H: (H - mu) / sd                                       # noqa: E731
    d_tr, d_te = z(Hc_tr) - z(Hr_tr), z(Hc_te) - z(Hr_te)
    l2, cv = cv_pick(d_tr)
    w = fit_head(d_tr, l2)
    out = {"feature": name, "l2_picked_by_cv": l2, "cv_accuracy": {str(k): v for k, v in cv.items()},
           "train_accuracy": round(pair_acc(d_tr, w), 4), "heldout_accuracy": round(pair_acc(d_te, w), 4)}
    print(f"  {name}: l2 {l2} (5-fold CV {cv[l2]:.3f}), train {out['train_accuracy']:.3f}, "
          f"held-out {out['heldout_accuracy']:.3f}")
    return out, (mu, sd, w)


@torch.no_grad()
def mps_vs_cpu(tok, text: tuple[str, str]) -> dict:
    """One full forward pass on MPS and on the CPU: logits and the layer-18 residual."""
    ids = torch.tensor([chat_ids(tok, *text)[0]])
    outs = {}
    for dev in ("mps", "cpu"):
        m = load_hf_model(dev)
        o = m(input_ids=ids.to(dev), output_hidden_states=True)
        outs[dev] = (o.logits[0].float().cpu(), o.hidden_states[LAYER][0].float().cpu())
        del m, o
        free()
    (lm, hm), (lc, hc) = outs["mps"], outs["cpu"]
    return {"tokens": int(ids.shape[1]),
            "max_abs_diff_logits": float((lm - lc).abs().max()),
            "logit_range": [round(float(lc.min()), 2), round(float(lc.max()), 2)],
            "max_abs_diff_layer18": float((hm - hc).abs().max()),
            "layer18_abs_max": round(float(hc.abs().max()), 2),
            "same_top1_every_position": bool((lm.argmax(-1) == lc.argmax(-1)).all())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("reward_head")
    torch.manual_seed(0)
    tok = load_tokenizer()
    pairs, info = data.load_pairs(tok)
    sp = data.splits(pairs)
    tr, te = sp["reward_train"], sp["heldout"]
    guard.tick("data")

    device = get_device()
    model = load_hf_model()
    texts = lambda ps, key: [(p["prompt"], p[key]) for p in ps]       # noqa: E731
    Lc_tr, Mc_tr = residuals(model, tok, texts(tr, "chosen"), device)
    guard.tick()
    Lr_tr, Mr_tr = residuals(model, tok, texts(tr, "rejected"), device)
    guard.tick()
    Lc_te, Mc_te = residuals(model, tok, texts(te, "chosen"), device)
    Lr_te, Mr_te = residuals(model, tok, texts(te, "rejected"), device)
    guard.tick()
    # The early stop must give what a full pass gives at hidden_states[18].
    ids = torch.tensor([chat_ids(tok, te[0]["prompt"], te[0]["chosen"])[0]], device=device)
    with torch.no_grad():
        full = model(input_ids=ids, output_hidden_states=True).hidden_states[LAYER][0, -1].float().cpu().numpy()
    early_stop_diff = float(np.abs(full - Lc_te[0]).max())
    del model
    free()
    guard.tick("features")

    last, (mu, sd, w) = run_feature("last token", Lc_tr, Lr_tr, Lc_te, Lr_te)
    mean, _ = run_feature("reply mean", Mc_tr, Mr_tr, Mc_te, Mr_te)

    zt = lambda H: ((H - mu) / sd) @ w                                # noqa: E731
    r_c, r_r = zt(Lc_te), zt(Lr_te)
    tok_c = np.array([p["reply_tokens_chosen"] for p in te])
    tok_r = np.array([p["reply_tokens_rejected"] for p in te])
    longer = np.where(tok_c != tok_r, (tok_c > tok_r).astype(float), 0.5)
    baselines = {
        "longer_reply_wins": round(float(longer.mean()), 4),
        "pairs_with_equal_length": int((tok_c == tok_r).sum()),
    }
    # How much of the head's reward is reply length? Correlation over all 1,000 held-out replies.
    r_all = np.concatenate([r_c, r_r])
    len_all = np.concatenate([tok_c, tok_r])
    corr = float(np.corrcoef(r_all, np.log(len_all))[0, 1])
    agree = float(((r_c > r_r) == (tok_c > tok_r))[tok_c != tok_r].mean())
    print(f"  longer reply wins: {baselines['longer_reply_wins']:.3f}; corr(reward, log length) {corr:.3f}; "
          f"head agrees with 'longer wins' on {agree:.3f} of pairs")
    guard.tick("heads")
    check = mps_vs_cpu(tok, (te[0]["prompt"], te[0]["chosen"]))
    print(f"  MPS vs CPU: logits {check['max_abs_diff_logits']:.2e}, layer 18 {check['max_abs_diff_layer18']:.2e}, "
          f"same top-1 {check['same_top1_every_position']}")
    guard.tick("mps_vs_cpu")

    res = {
        "model": "HuggingFaceTB/SmolLM2-360M-Instruct", "layer": LAYER, "hidden_size": int(len(w)),
        "data": {"dataset": info["dataset"], "revision": info["revision"], "train_pairs": len(tr),
                 "heldout_pairs": len(te), "max_tokens": info["max_tokens"]},
        "early_stop_vs_full_pass_max_abs_diff": early_stop_diff,
        "heads": {"last_token": last, "reply_mean": mean}, "main_feature": "last_token",
        "baselines": baselines,
        "length": {"corr_reward_log_tokens_heldout_replies": round(corr, 4),
                   "head_agrees_with_longer_wins": round(agree, 4)},
        "heldout_scores": [{"id": p["id"], "r_chosen": round(float(a), 4), "r_rejected": round(float(b), 4),
                            "tokens_chosen": int(c), "tokens_rejected": int(d)}
                           for p, a, b, c, d in zip(te, r_c, r_r, tok_c, tok_r)],
        "head": {"mu": [round(float(x), 5) for x in mu], "sd": [round(float(x), 5) for x in sd],
                 "w": [round(float(x), 6) for x in w]},
        "mps_vs_cpu": check,
        "guard": guard.report(),
    }
    write_results("reward_head", res)
    if args.figures:
        fig_reward(res, FIGURES / "reward_head.png")


def fig_reward(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    names = ["chance", "longer reply\nwins", "head:\nreply mean", "head:\nlast token"]
    vals = [0.5, r["baselines"]["longer_reply_wins"], r["heads"]["reply_mean"]["heldout_accuracy"],
            r["heads"]["last_token"]["heldout_accuracy"]]
    cols = ["#cccccc", "#888888", "#e7a07f", "#b7410e"]
    a.bar(names, vals, color=cols)
    for k, v in enumerate(vals):
        a.text(k, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    a.set_ylim(0, 1)
    a.set_ylabel("held-out pair accuracy")
    a.set_title(f"{r['data']['heldout_pairs']} held-out pairs", fontsize=10)
    s = r["heldout_scores"]
    rc = np.array([x["r_chosen"] for x in s]); rr = np.array([x["r_rejected"] for x in s])
    tc = np.array([x["tokens_chosen"] for x in s]); tr = np.array([x["tokens_rejected"] for x in s])
    b.scatter(np.log(tc / tr), rc - rr, s=8, alpha=0.5, color="#b7410e")
    b.axhline(0, color="#999999", lw=0.8)
    b.axvline(0, color="#999999", lw=0.8)
    b.set_xlabel("log(chosen tokens / rejected tokens)")
    b.set_ylabel("r(chosen) − r(rejected)")
    b.set_title("Reward margin against length ratio", fontsize=10)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
