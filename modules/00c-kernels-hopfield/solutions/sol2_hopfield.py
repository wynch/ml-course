"""Solution to exercise 2 — Hebbian storage, asynchronous recall, capacity.

The measured threshold at these sizes lands *above* the asymptotic 0.138 and
drifts down as N grows — a finite-size effect, not a bug.

Run:  cd python && uv run ../solutions/sol2_hopfield.py
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
    P = np.asarray(patterns, dtype=float)
    W = (P.T @ P) / P.shape[1]
    np.fill_diagonal(W, 0.0)
    return W


def async_update(W, s, rng, sweeps=8):
    """Update one neuron at a time in random order until nothing changes."""
    s = np.array(s, dtype=float).copy()
    for _ in range(sweeps):
        changed = False
        for i in rng.permutation(len(s)):
            h = float(W[i] @ s)
            new = 1.0 if h >= 0.0 else -1.0
            if new != s[i]:
                s[i] = new
                changed = True
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
            pats = rng.choice([-1.0, 1.0], size=(p, n))
            W = hebbian_weights(pats)
            mu = int(rng.integers(p))
            got = async_update(W, corrupt(pats[mu], noise, rng), rng)
            ok += abs(overlap(got, pats[mu])) >= 0.95
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
