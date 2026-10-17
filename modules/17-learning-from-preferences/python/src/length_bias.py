"""Section 4: length bias, in the data and in the replies.

No model is loaded. It reads the cached pairs and the results of sections 1
and 3:

- in the data: how often the chosen reply is the longer one, over every
  single-turn pair read (before our 640-token filter) and by rating gap;
- a reward that reads only the reply's length, trained like the head;
- in the reward head: its held-out accuracy when the chosen reply is the
  longer one, and when it is the shorter one;
- in DPO: the same split for the implicit reward after training, and the
  length of the generated replies before and after.

    uv run python src/length_bias.py --figures
"""

from __future__ import annotations

import argparse

from common import FIGURES, Guard, read_results, write_results

import numpy as np

import data

GAP_BINS = ((0, 1), (1, 2), (2, 3), (3, 10))


def words(s: str) -> int:
    return len(s.split())


def split_acc(correct: np.ndarray, lc: np.ndarray, lr: np.ndarray) -> dict:
    longer, shorter = lc > lr, lc < lr
    return {"chosen_longer": {"pairs": int(longer.sum()), "accuracy": round(float(correct[longer].mean()), 4)},
            "chosen_shorter": {"pairs": int(shorter.sum()), "accuracy": round(float(correct[shorter].mean()), 4)},
            "all": {"pairs": len(correct), "accuracy": round(float(correct.mean()), 4)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("length_bias")

    raw, _ = data.fetch_raw()
    allp = [p for s in ("train", "test") for p in raw[s] if p["score_chosen"] != p["score_rejected"]]
    wc = np.array([words(p["chosen"]) for p in allp])
    wr = np.array([words(p["rejected"]) for p in allp])
    gap = np.array([p["score_chosen"] - p["score_rejected"] for p in allp])
    by_gap = []
    for lo, hi in GAP_BINS:
        m = (gap > lo) & (gap <= hi) if lo else (gap > 0) & (gap <= hi)
        by_gap.append({"gap": [lo, hi], "pairs": int(m.sum()),
                       "chosen_longer": round(float((wc[m] > wr[m]).mean()), 4),
                       "median_word_ratio": round(float(np.median(wc[m] / np.maximum(wr[m], 1))), 3)})
    raw_stats = {"pairs": len(allp), "ties_dropped": sum(p["score_chosen"] == p["score_rejected"]
                                                         for s in ("train", "test") for p in raw[s]),
                 "chosen_longer": round(float((wc > wr).mean()), 4),
                 "equal": round(float((wc == wr).mean()), 4),
                 "median_words_chosen": float(np.median(wc)), "median_words_rejected": float(np.median(wr)),
                 "by_rating_gap": by_gap}
    pairs, _ = data.load_pairs()
    sp = data.splits(pairs)
    kept = {}
    for name, ps in sp.items():
        tc = np.array([p["reply_tokens_chosen"] for p in ps]); tr = np.array([p["reply_tokens_rejected"] for p in ps])
        kept[name] = {"pairs": len(ps), "chosen_longer": round(float((tc > tr).mean()), 4),
                      "median_tokens_chosen": float(np.median(tc)), "median_tokens_rejected": float(np.median(tr))}
    # The DPO run trains only on pairs that fit TRAIN_MAX_TOKENS (dpo_train.py); read the cap from its
    # results so this script stays free of torch.
    try:
        cap = read_results("dpo_full")["settings"]["train_max_tokens"]
        ps = [p for p in sp["dpo_train"] if max(p["len_chosen"], p["len_rejected"]) <= cap]
        tc = np.array([p["reply_tokens_chosen"] for p in ps]); tr = np.array([p["reply_tokens_rejected"] for p in ps])
        kept["dpo_train_used"] = {"pairs": len(ps), "max_tokens": cap, "chosen_longer": round(float((tc > tr).mean()), 4),
                                  "median_tokens_chosen": float(np.median(tc)), "median_tokens_rejected": float(np.median(tr))}
    except FileNotFoundError:
        pass
    print(f"data: chosen longer in {raw_stats['chosen_longer']:.3f} of {raw_stats['pairs']} pairs "
          f"(before the 640-token filter); after it, held-out {kept['heldout']['chosen_longer']:.3f}")

    # A reward that reads only the log of the reply's length, trained like the head.
    # A reply can be empty (0 tokens), so lengths count as at least 1.
    ld = lambda ps: (np.log(np.maximum([p["reply_tokens_chosen"] for p in ps], 1))      # noqa: E731
                     - np.log(np.maximum([p["reply_tokens_rejected"] for p in ps], 1)))
    d_tr, d_te = ld(sp["reward_train"]), ld(sp["heldout"])
    w = 0.0
    for _ in range(2000):
        w -= 0.5 * float(np.mean(-(1 - 1 / (1 + np.exp(-w * d_tr))) * d_tr))
    length_only = {"w": round(w, 4), "train_pairs": len(d_tr), "heldout_pairs": len(d_te),
                   "heldout_accuracy": round(float(np.mean(w * d_te > 0) + 0.5 * np.mean(d_te == 0)), 4)}
    print(f"length-only reward: w {w:.3f}, held-out accuracy {length_only['heldout_accuracy']:.3f}")

    rh = read_results("reward_head")["heldout_scores"]
    rc = np.array([x["r_chosen"] for x in rh]); rr = np.array([x["r_rejected"] for x in rh])
    lc = np.array([x["tokens_chosen"] for x in rh]); lr = np.array([x["tokens_rejected"] for x in rh])
    head = split_acc(rc > rr, lc, lr)
    print(f"reward head: {head['chosen_longer']['accuracy']:.3f} when chosen is longer, "
          f"{head['chosen_shorter']['accuracy']:.3f} when shorter")

    dpo = {}
    for cfg in ("full", "preview"):
        try:
            d = read_results(f"dpo_{cfg}")
        except FileNotFoundError:
            continue
        pp = d["heldout"]["per_pair"]
        m = np.array([x["margin"] for x in pp]); rd = np.array([x["ref_logp_diff"] for x in pp])
        a = np.array([x["tokens_chosen"] for x in pp]); b = np.array([x["tokens_rejected"] for x in pp])
        g = d["generations"]
        dpo[cfg] = {"implicit_reward_after": split_acc(m > 0, a, b),
                    "reference_logp": split_acc(rd > 0, a, b),
                    "reply_tokens_mean": [g["before"]["tokens_mean"], g["after"]["tokens_mean"]],
                    "reply_tokens_median": [g["before"]["tokens_median"], g["after"]["tokens_median"]],
                    "replies_cut_at_max": [g["prompts"] - g["before"]["finished"], g["prompts"] - g["after"]["finished"]],
                    "longer_after": sum(s["tokens"][1] > s["tokens"][0] for s in g["samples"]),
                    "shorter_after": sum(s["tokens"][1] < s["tokens"][0] for s in g["samples"]),
                    "prompts": g["prompts"]}
        x = dpo[cfg]["implicit_reward_after"]
        print(f"DPO {cfg}: implicit reward {x['chosen_longer']['accuracy']:.3f} when chosen is longer, "
              f"{x['chosen_shorter']['accuracy']:.3f} when shorter; reply tokens "
              f"{dpo[cfg]['reply_tokens_mean'][0]} -> {dpo[cfg]['reply_tokens_mean'][1]}")

    res = {"data_raw": raw_stats, "data_kept": kept, "length_only_reward": length_only, "reward_head": head, "dpo": dpo, "guard": guard.report()}
    write_results("length_bias", res)
    if args.figures:
        fig_length(res, FIGURES / "length_bias.png")


def fig_length(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    bg = r["data_raw"]["by_rating_gap"]
    a.bar([f"{x['gap'][0]}–{x['gap'][1]}\n({x['pairs']})" for x in bg], [x["chosen_longer"] for x in bg],
          color="#b7410e")
    a.axhline(0.5, color="#999999", lw=0.8, ls="--")
    a.set_ylim(0, 1)
    a.set_xlabel("rating gap, chosen − rejected (pairs)")
    a.set_ylabel("share where chosen is longer")
    a.set_title("In the data", fontsize=10)
    groups = [("reward head", r["reward_head"])]
    if "full" in r["dpo"]:
        groups += [("reference log-prob", r["dpo"]["full"]["reference_logp"]),
                   ("DPO implicit reward", r["dpo"]["full"]["implicit_reward_after"])]
    x = np.arange(len(groups))
    b.bar(x - 0.18, [g["chosen_longer"]["accuracy"] for _, g in groups], 0.36, color="#b7410e",
          label="chosen is longer")
    b.bar(x + 0.18, [g["chosen_shorter"]["accuracy"] for _, g in groups], 0.36, color="#888888",
          label="chosen is shorter")
    b.set_xticks(x, [n for n, _ in groups])
    b.axhline(0.5, color="#999999", lw=0.8, ls="--")
    b.set_ylim(0, 1)
    b.set_ylabel("held-out pair accuracy")
    b.set_title("Accuracy split by which reply is longer", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
