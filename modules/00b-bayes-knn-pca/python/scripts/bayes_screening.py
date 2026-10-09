"""The screening-test worked example, printed and plotted.

A fictional condition and a fictional test, chosen so the arithmetic is clean:

    prevalence  P(D)          = 0.004      (4 in 1000)
    sensitivity P(+ | D)      = 0.96
    specificity P(- | not D)  = 0.92   ->  false-positive rate 0.08

The likelihood ratio of a positive result is 0.96 / 0.08 = 12 exactly, so each
independent positive multiplies the odds by 12 and the whole story fits on one
line of arithmetic.

Produces: figures/bayes_posterior.png

Run:  uv run scripts/bayes_screening.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import numpy as np

from origins.bayes import likelihood_ratio, odds, posterior, prob, sequential_posteriors
from origins.plots import ACCENT, MUTED, T0, WARN, finish, plt

PREV = 0.004
SENS = 0.96
SPEC = 0.92
N_TESTS = 4


def main() -> None:
    lr_pos = likelihood_ratio(SENS, SPEC, positive=True)
    lr_neg = likelihood_ratio(SENS, SPEC, positive=False)
    print("Screening example — prevalence 0.4%, sensitivity 96%, specificity 92%")
    print(f"  LR+ = {SENS:.2f} / {1 - SPEC:.2f} = {lr_pos:.4f}")
    print(f"  LR- = {1 - SENS:.2f} / {SPEC:.2f} = {lr_neg:.6f}")
    print(f"  prior odds = {odds(PREV):.6f}  (1 in {1 / odds(PREV):.0f})")

    post = sequential_posteriors(PREV, SENS, SPEC, [True] * N_TESTS)
    chain = [PREV] + post
    for i, p in enumerate(chain):
        print(f"  after {i} positive test(s): P(D | evidence) = {p:.6f}  ({100 * p:.2f}%)")

    # the single-test numbers spelled out, so the README can quote them
    joint_pos = PREV * SENS + (1 - PREV) * (1 - SPEC)
    print(f"  P(+) = {PREV}*{SENS} + {1 - PREV:.3f}*{1 - SPEC:.2f} = {joint_pos:.6f}")
    print(f"  P(D | -) = {posterior(PREV, SENS, SPEC, positive=False):.8f}")

    # cross-check: sequential updating == one-shot Bayes with LR+^k
    for k in range(1, N_TESTS + 1):
        one_shot = prob(odds(PREV) * lr_pos ** k)
        assert abs(one_shot - chain[k]) < 1e-12, (k, one_shot, chain[k])
    print("  check: sequential updating == prior odds x LR+^k  (max err < 1e-12)")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    # (a) posterior after k positive tests
    ax = axes[0]
    xs = np.arange(len(chain))
    bars = ax.bar(xs, [100 * p for p in chain], color=[MUTED] + [T0] * N_TESTS, width=0.62)
    for x, p in zip(xs, chain):
        ax.text(x, 100 * p + 2.5, f"{100 * p:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["prior"] + [f"+{i}" for i in xs[1:]])
    ax.set_xlabel("positive results so far")
    ax.set_ylabel("P(condition | evidence)   %")
    ax.set_ylim(0, 112)
    ax.set_title("(a) one test proves almost nothing")
    bars[1].set_edgecolor(WARN)
    bars[1].set_linewidth(1.6)

    # (b) the same thing in log-odds, where Bayes is a straight line
    ax = axes[1]
    lo = [np.log10(odds(p)) for p in chain]
    ax.plot(xs, lo, "o-", color=T0, lw=1.8, ms=5)
    ax.axhline(0.0, color=WARN, lw=1.0, ls="--")
    ax.text(0.05, 0.06, "log-odds 0  =  50/50", color=WARN, fontsize=8)
    step = np.log10(lr_pos)
    ax.annotate(
        f"every positive adds\nlog10(LR+) = {step:.3f}",
        xy=(2, lo[2]), xytext=(0.35, 1.05),
        fontsize=8, color=MUTED,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
    )
    ax.set_xticks(xs)
    ax.set_xlabel("positive results so far")
    ax.set_ylabel("log10 odds")
    ax.set_title("(b) in odds space it is a straight line")

    # (c) the posterior after ONE positive, as prevalence varies
    ax = axes[2]
    prevs = np.logspace(-4, -0.3, 300)
    posts = [100 * posterior(p, SENS, SPEC) for p in prevs]
    ax.semilogx(prevs, posts, color=ACCENT, lw=1.8)
    ax.plot([PREV], [100 * chain[1]], "o", color=WARN, ms=6)
    ax.annotate(
        f"prevalence {PREV:.3f}\n-> {100 * chain[1]:.1f}%",
        xy=(PREV, 100 * chain[1]), xytext=(6e-4, 55),
        fontsize=8, color=WARN,
        arrowprops=dict(arrowstyle="->", color=WARN, lw=0.9),
    )
    ax.set_xlabel("prevalence P(condition)")
    ax.set_ylabel("P(condition | one positive)   %")
    ax.set_ylim(0, 100)
    ax.set_title("(c) the same test, any base rate")

    fig.tight_layout()
    finish(fig, FIGDIR / "bayes_posterior.png")


if __name__ == "__main__":
    main()
