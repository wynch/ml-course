"""Exercise 2 — build the memory, then find out how much it holds.

Three TODOs:

  1. ``hebbian_weights`` — the storage rule, W = (1/N) Σ ξξᵀ with W_ii = 0.
  2. ``async_update``    — recall: sweep the neurons in random order and set
                           each one to sign of its local field (Ws)ᵢ.
  3. ``capacity``        — sweep the load α = P/N with random patterns and find
                           where recall collapses. Compare with 0.138·N.

Once all three work the script prints a capacity table. The measured threshold
at these sizes will land *above* 0.138 — that is a finite-size effect, and the
last part of the script shows it shrinking as N grows.

Run:  cd python && uv run ../exercises/ex2_hopfield.py
"""

import numpy as np

GLYPHS = {  # 10×10 letters, flattened row-major; '#' is +1, '.' is −1
    "A": ("..######...##....##.##......####......############"
          "##......####......####......####......####......##"),
    "T": ("####################....##........##........##...."
          "....##........##........##........##........##...."),
    "L": ("##........##........##........##........##........"
          "##........##........##........####################"),
}


def patterns_from_glyphs():
    return np.asarray([[1.0 if ch == "#" else -1.0 for ch in art]
                       for art in GLYPHS.values()])


def hebbian_weights(patterns):
    """W = (1/N) Σ_μ ξ^μ (ξ^μ)ᵀ with a zero diagonal. ``patterns`` is (P, N)."""
    # TODO(you): two lines — one outer-product sum, one diagonal wipe.
    #   (np.fill_diagonal mutates in place; the sum is a single matrix product.)
    raise NotImplementedError


def async_update(W, s, rng, sweeps=8):
    """Update one neuron at a time in random order until nothing changes."""
    s = np.array(s, dtype=float).copy()
    for _ in range(sweeps):
        changed = False
        for i in rng.permutation(len(s)):
            # TODO(you): compute the local field h = (Ws)_i, set s[i] to +1 if
            # h >= 0 else −1, and remember whether the value actually changed.
            raise NotImplementedError
        if not changed:
            break
    return s


def capacity(n, alphas, trials=25, noise=0.05, seed=11):
    """Fraction of patterns recalled (overlap ≥ 0.95) at each load α = P/N."""
    rng = np.random.default_rng(seed)
    out = []
    for a in alphas:
        p = max(1, int(round(a * n)))
        ok = 0
        for _ in range(trials):
            # TODO(you): draw p random ±1 patterns of length n, store them,
            # corrupt one of them by flipping `noise` of its bits, recall it,
            # and count a success when the overlap with the target is ≥ 0.95.
            raise NotImplementedError
        out.append(ok / trials)
    return np.asarray(out)


# ─────────────────────────────────────────── everything below is provided ──

def overlap(a, b):
    return float(np.asarray(a) @ np.asarray(b) / len(a))


def corrupt(pattern, frac, rng):
    s = np.array(pattern, dtype=float).copy()
    idx = rng.choice(len(s), size=int(round(frac * len(s))), replace=False)
    s[idx] *= -1.0
    return s


def crossing(alphas, frac, level=0.5):
    for i in range(len(alphas) - 1):
        if frac[i] >= level > frac[i + 1]:
            t = (frac[i] - level) / (frac[i] - frac[i + 1])
            return float(alphas[i] + t * (alphas[i + 1] - alphas[i]))
    return float("nan")


def main():
    pats = patterns_from_glyphs()
    W = hebbian_weights(pats)
    rng = np.random.default_rng(0)

    print("part 1 — three letters, 30% of the bits destroyed")
    repaired = 0
    for name, pat in zip(GLYPHS, pats):
        got = async_update(W, corrupt(pat, 0.30, rng), rng)
        m = overlap(got, pat)
        repaired += m >= 0.95
        print(f"  {name}: overlap after recall {m:+.3f}")

    print("\npart 2 — capacity of a random-pattern memory")
    alphas = np.round(np.arange(0.04, 0.281, 0.02), 4)
    rows = []
    for n in (100, 400, 1600):
        frac = capacity(n, alphas)
        ac = crossing(alphas, frac)
        rows.append(ac)
        print(f"  N = {n:5d}   collapse at alpha = {ac:.3f}   "
              f"(0.138·N = {0.138 * n:6.1f} patterns, measured {ac * n:6.1f})")

    checks = [
        ("all three letters repaired", repaired == 3),
        ("W is symmetric", bool(np.allclose(W, W.T))),
        ("W has no self-coupling", bool(np.allclose(np.diag(W), 0))),
        ("collapse is above the asymptotic 0.138", all(a > 0.138 for a in rows)),
        ("collapse drifts down as N grows", rows[0] > rows[1] > rows[2]),
        ("and is heading for 0.138, not 0.5", rows[-1] < 0.20),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\nall good" if all(ok for _n, ok in checks) else "\nnot yet")


if __name__ == "__main__":
    main()
