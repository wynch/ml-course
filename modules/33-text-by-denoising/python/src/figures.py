"""Redraw every figure of module 33 from results/*.json; no model is loaded.

    uv run python src/figures.py
"""

from __future__ import annotations

import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

COL = {"mdm6x": "#1d6fb8", "fm": "#1d6fb8", "ddpm": "#b7410e", "ddim": "#d9a63a", "ar": "#3a3a38", "mdm": "#3fa34d",
       "bd4": "#7b4fb0", "bd16": "#1f918d", "random": "#d1495b", "confidence": "#3fa34d", "ard3": "#8a8f8d",
       "data": "#8a8f8d"}
STEP_LABEL = {"fm": "flow matching, Euler", "ddpm": "DDPM, ancestral (module 09)", "ddim": "DDPM, DDIM"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def clean_log(log: list[dict]) -> list[dict]:
    """The committed run logs were written before train.py carried the running sum across
    chunks, so the first 50-step average after each chunk boundary covers fewer than 50 steps
    and reads low. Drop any average more than 3% below the median of the five around it."""
    v = np.array([r["loss"] for r in log])
    keep = [v[i] >= 0.97 * np.median(v[max(0, i - 2): i + 3]) for i in range(len(v))]
    return [r for r, k in zip(log, keep) if k]


def flow() -> None:
    R, P = common.read_results("flow2d"), common.read_results("flow2d_points")
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1])
    held = np.array(P["held_out"])
    for j, n in enumerate((1, 4, 64)):
        for i, kind in enumerate(("fm", "ddpm")):
            ax = fig.add_subplot(gs[i, j])
            ax.scatter(held[:, 0], held[:, 1], s=2, c=COL["data"], alpha=0.25)
            pts = np.array(P[f"{kind}_{n}"])
            ax.scatter(pts[:, 0], pts[:, 1], s=3, c=COL[kind])
            w = R["results"][kind][str(n)]["sliced_w1"]
            ax.set_title(f"{STEP_LABEL[kind]}\n{n} step{'s' if n > 1 else ''}: sliced W1 {w}", fontsize=9)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 3:])
    steps = R["steps"]
    for kind in ("fm", "ddpm", "ddim"):
        ax.plot(steps, [R["results"][kind][str(n)]["sliced_w1"] for n in steps], "o-", color=COL[kind], label=STEP_LABEL[kind])
    ax.axhline(R["eval"]["floor_two_spiral_draws"], color=COL["data"], ls="--", lw=1, label="two draws of the spiral (floor)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sampling steps")
    ax.set_ylabel("sliced W1 to held-out points (lower = closer)")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    ax = fig.add_subplot(gs[1, 3:])
    for kind in ("fm", "ddpm"):
        paths = np.array(P[f"{kind}_paths"])
        for p in paths[:6]:
            ax.plot(p[:, 0], p[:, 1], color=COL[kind], lw=0.8, alpha=0.8)
            ax.plot(p[-1, 0], p[-1, 1], "o", ms=2.5, color=COL[kind])
    ax.scatter(held[:, 0], held[:, 1], s=2, c=COL["data"], alpha=0.2)
    ax.set_title("6 paths from the same noise, 64 steps: flow matching (blue) and DDPM ancestral (red)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle("Section 1: from noise to the spiral. One step lands on the average; flow matching gets there in fewer steps",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "flow.png", dpi=120)
    plt.close(fig)


def likelihood() -> None:
    E = common.read_results("evaluate")
    L = E["likelihood_nats_per_char"]
    runs = {a: common.read_results(f"runs/{a}") for a in ("ar", "mdm", "bd4", "bd16", "mdm6x")}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})
    for a in ("ar", "mdm", "bd4", "bd16", "mdm6x"):
        log = clean_log(runs[a]["loss_log"])
        ax[0].plot([r["step"] for r in log], [r["loss"] for r in log], color=COL[a], lw=1, label=a)
    ax[0].set_ylim(1.4, 3.4)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("training step (log)")
    ax[0].set_ylabel("training loss (AR: exact; diffusion: the bound), nats/char")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Same network, same recipe; only the loss differs", fontsize=9)
    rows = [("AR, exact", L["ar_exact"], COL["ar"]), ("bd4, bound", L["bd4_elbo"], COL["bd4"]),
            ("bd16, bound", L["bd16_elbo"], COL["bd16"]), ("mdm, bound", L["mdm_elbo"], COL["mdm"]),
            ("mdm, exact left to right", L["mdm_exact_l2r"], COL["mdm"]),
            ("mdm, exact right to left", L["mdm_exact_r2l"], COL["mdm"]),
            ("mdm, exact random order", L["mdm_exact_random"], COL["mdm"]),
            ("mdm6x, bound", L["mdm6x_elbo"], COL["fm"]), ("mdm6x, exact left to right", L["mdm6x_exact_l2r"], COL["fm"])]
    y = np.arange(len(rows))[::-1]
    ax[1].barh(y, [r[1] for r in rows], color=[r[2] for r in rows], alpha=0.85)
    for yi, r in zip(y, rows):
        ax[1].text(r[1] + 0.01, yi, f"{r[1]:.3f}", va="center", fontsize=8)
    ax[1].set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax[1].set_xlim(1.5, max(r[1] for r in rows) + 0.15)
    ax[1].set_xlabel("held-out NLL per character, nats (lower = better)")
    ax[1].set_title("256 held-out blocks of 128 characters", fontsize=9)
    F = E["infill"]["results"]
    a0, b0 = E["infill"]["span"]
    pos = list(range(a0, b0))
    for k, lab, c, ls in (("ar_left_only", "AR, left side only", COL["ar"], "-"),
                          ("mdm6x_left_only", "mdm6x, left side only", COL["fm"], ":"),
                          ("mdm6x_both_sides", "mdm6x, both sides", COL["fm"], "-"),
                          ("mdm_both_sides", "mdm, both sides", COL["mdm"], "-")):
        ax[2].plot(pos, F[k]["by_position"], ls, marker="o", ms=3, color=c,
                   label=f"{lab}: {F[k]['char_accuracy']:.3f} overall")
    ax[2].set_xticks(pos[::3])
    ax[2].set_xlabel("position of the hidden character in the block")
    ax[2].set_ylabel("share right, greedy")
    ax[2].set_ylim(0, 0.6)
    ax[2].legend(frameon=False, fontsize=7.5)
    ax[2].set_title(f"Fill characters {a0}-{b0 - 1} of each held-out block", fontsize=9)
    for a in ax:
        _style(a)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "likelihood.png", dpi=120)
    plt.close(fig)


def steps() -> None:
    S = common.read_results("sample")
    st = S["steps"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for k, (key, lab) in enumerate((("word_score", "share of words found in the training text"),
                                    ("judge_nll", "AR judge's NLL per character (lower = likelier)"),
                                    ("distinct4", "distinct 4-grams / all 4-grams"))):
        for name, ls in (("mdm", "o-"), ("mdm6x", "s--")):
            for rule in ("random", "confidence"):
                ax[k].plot(st, [S["results"][name][rule][str(n)][key] for n in st], ls, color=COL[rule], ms=4,
                           label=f"{name}, {rule}")
        ax[k].axhline(S["references"]["ar_samples"][key], color=COL["ar"], ls="--", lw=1, label="AR twin's own samples")
        ax[k].axhline(S["references"]["held_out_text"][key], color=COL["data"], ls=":", lw=1.2, label="held-out text")
        ax[k].set_xscale("log", base=2)
        ax[k].set_xticks(st, [str(n) for n in st])
        ax[k].set_xlabel("steps to write 128 characters")
        ax[k].set_ylabel(lab, fontsize=9)
        _style(ax[k])
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Section 3: 64 lines per point, temperature 1 (circles: mdm; squares, dashed: mdm6x)", fontsize=11)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "steps.png", dpi=120)
    plt.close(fig)


def block() -> None:
    E, B = common.read_results("evaluate"), common.read_results("block")
    L = E["likelihood_nats_per_char"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    xs, ys = [1, 4, 16, 128], [L["ar_exact"], L["bd4_elbo"], L["bd16_elbo"], L["mdm_elbo"]]
    ax[0].plot(xs, ys, "o-", color=COL["bd16"])
    for x, y, n in zip(xs, ys, ("AR (exact)", "bd4", "bd16", "mdm")):
        ax[0].annotate(f"{n}\n{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(xs, [str(x) for x in xs])
    ax[0].set_xlabel("block length L")
    ax[0].set_ylabel("held-out NLL bound, nats/char")
    ax[0].set_ylim(min(ys) - 0.1, max(ys) + 0.15)
    ax[0].set_title("Smaller blocks, tighter bound", fontsize=9)
    R = B["results"]["128"]
    keys = [("ar_cache", "AR, cache"), ("ar_nocache", "AR, no cache"), ("bd4_cache", "bd4, cache"),
            ("bd4_nocache", "bd4, no cache"), ("bd16_cache", "bd16, cache"), ("bd16_nocache", "bd16, no cache"),
            ("mdm", "mdm")]
    cols = [COL[k.split("_")[0]] for k, _ in keys]
    ax[1].barh(range(len(keys))[::-1], [R[k]["positions"] for k, _ in keys], color=cols)
    ax[1].set_yticks(range(len(keys))[::-1], [n for _, n in keys], fontsize=8)
    ax[1].set_xlabel("positions run through the network per line")
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    ax[1].set_title("One character per step (128 steps)", fontsize=9)
    ax[2].barh(range(len(keys))[::-1], [R[k]["seconds_for_64"] for k, _ in keys], color=cols)
    for i, (k, _) in enumerate(keys):
        ax[2].text(R[k]["seconds_for_64"], len(keys) - 1 - i, f" {R[k]['seconds_for_64']:.2f} s · words {R[k]['word_score']:.2f}",
                   va="center", fontsize=7.5)
    ax[2].set_yticks(range(len(keys))[::-1], [n for _, n in keys], fontsize=8)
    ax[2].set_xlabel("seconds for 64 lines on MPS (median of 3)")
    ax[2].set_xlim(0, max(R[k]["seconds_for_64"] for k, _ in keys) * 1.6)
    ax[2].set_title("Wall time, and the share of real words", fontsize=9)
    for a in ax:
        _style(a)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "block.png", dpi=120)
    plt.close(fig)


def repeat() -> None:
    R = common.read_results("repeat")["sizes"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for k, size in enumerate(("r100k", "r1m")):
        if size not in R:
            continue
        S = R[size]
        for name, lab in (("ar", "AR, dropout 0.1 (exact)"), ("ard3", "AR, dropout 0.3 (exact)"), ("mdm", "mdm (bound)")):
            if name not in S:
                continue
            a = S[name]
            ax[k].plot(a["epochs"], a["val_mean"], "o-", color=COL[name], label=lab)
            if len(a["seeds"]) > 1:
                ax[k].fill_between(a["epochs"], a["val_min_seed"], a["val_max_seed"], color=COL[name], alpha=0.15, lw=0)
            ax[k].plot(a["epochs"], a["train_mean"], ":", color=COL[name], lw=1)
        m = S["mdm"]
        ax[k].plot(m["epochs"], m["val_l2r_mean"], "s--", color=COL["mdm"], ms=3, lw=1, label="mdm, exact left to right")
        axis = m["epochs"]
        if "mdmlong" in S:
            lg = S["mdmlong"]
            ax[k].plot(lg["epochs"], lg["val_mean"], "-", color=COL["mdm"], lw=0.9, label="mdm, seed 0 on to 800 epochs (bound)")
            axis = lg["epochs"]
        ax[k].set_xscale("log", base=2)
        ax[k].set_xticks(axis, [str(e) for e in axis])
        ax[k].set_xlabel("epochs (characters drawn / unique characters)")
        ax[k].set_ylabel("NLL per character, nats; dotted = on its own training text")
        ax[k].set_title(f"{S['unique_chars']:,} unique characters, seeds {len(S['mdm']['seeds'])}", fontsize=9)
        ax[k].set_ylim(0.8, 3.9)
        _style(ax[k])
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Section 5: the same small data, seen again and again", fontsize=11)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "repeat.png", dpi=120)
    plt.close(fig)


def main() -> None:
    common.FIGURES.mkdir(exist_ok=True)
    for f in (flow, likelihood, steps, block, repeat):
        try:
            f()
            print(f"  drew {f.__name__}")
        except FileNotFoundError as e:
            print(f"  skipped {f.__name__}: {e.filename} missing")


if __name__ == "__main__":
    main()
