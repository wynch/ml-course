"""Hopfield networks: memory as the bottom of a valley.

A Hopfield network is N binary neurons, sᵢ ∈ {−1, +1}, wired symmetrically to
each other with no self-connections. Storing a set of patterns {ξ¹ … ξᴾ} is a
single outer-product sum — no iteration, no gradient:

    W = (1/N) Σ_μ ξ^μ (ξ^μ)ᵀ ,     W_ii = 0.

Recall is a physics problem. Define the energy

    E(s) = −½ sᵀWs

and update one neuron at a time, sᵢ ← sign((Ws)ᵢ). Each such flip can only
*lower* E — the change is ΔE = −Δsᵢ·(Ws)ᵢ, and the update chooses the sign of
sᵢ that makes that product positive — so the state slides downhill and stops.
Stored patterns are (nearly) the local minima, which is why a corrupted input
rolls into the clean original. That is content-addressable memory: you address
the store with a piece of the content rather than an index.

The catch is capacity. Cram in too many patterns and their crosstalk digs
spurious valleys; the classic result is a sharp collapse a little above
P ≈ 0.138·N. :func:`capacity_curve` measures it instead of quoting it.

:func:`modern_update` is the 2020 continuous version, which is where this
module meets module 04: it is literally one step of softmax attention.
"""

from __future__ import annotations

import numpy as np

# ───────────────────────────────────────────────────── classical Hopfield ──


def hebbian_weights(patterns: np.ndarray) -> np.ndarray:
    """W = (1/N) Σ ξξᵀ with a zero diagonal. ``patterns`` is (P, N) in ±1."""
    P = np.asarray(patterns, dtype=float)
    n = P.shape[1]
    W = (P.T @ P) / n
    np.fill_diagonal(W, 0.0)
    return W


def energy(W: np.ndarray, s: np.ndarray) -> float:
    """E(s) = −½ sᵀWs. Every accepted asynchronous flip strictly decreases it."""
    s = np.asarray(s, dtype=float)
    return float(-0.5 * s @ W @ s)


def async_update(W: np.ndarray, s: np.ndarray, rng: np.random.Generator,
                 sweeps: int = 8, record: bool = False):
    """Asynchronous recall: sweep the neurons in random order, one flip at a time.

    Returns ``(final_state, trace)`` where ``trace`` — only filled when
    ``record`` is true — is a list of ``(neuron_updates_done, energy, state)``
    sampled after every neuron update. Stops early once a full sweep changes
    nothing, which is the definition of a fixed point.
    """
    s = np.array(s, dtype=float).copy()
    n = len(s)
    trace = [(0, energy(W, s), s.copy())] if record else []
    done = 0
    for _ in range(sweeps):
        changed = False
        for i in rng.permutation(n):
            h = float(W[i] @ s)
            new = 1.0 if h >= 0.0 else -1.0
            if new != s[i]:
                s[i] = new
                changed = True
            done += 1
            if record:
                trace.append((done, energy(W, s), s.copy()))
        if not changed:
            break
    return s, trace


def corrupt(pattern: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Flip a fraction of the bits — the "damaged input" you ask the net to fix."""
    s = np.array(pattern, dtype=float).copy()
    k = int(round(frac * len(s)))
    idx = rng.choice(len(s), size=k, replace=False)
    s[idx] *= -1.0
    return s


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """m = (1/N)·aᵀb ∈ [−1, 1]. 1.0 means "recalled exactly"."""
    return float(np.asarray(a, dtype=float) @ np.asarray(b, dtype=float) / len(a))


# ─────────────────────────────────────────────────────────────── capacity ──


def capacity_curve(n: int = 400, alphas=None, trials: int = 20, noise: float = 0.05,
                   sweeps: int = 8, seed: int = 11):
    """Measure recall quality as a function of the load α = P/N.

    For each α: draw P = round(αN) *random* ±1 patterns, store them Hebbianly,
    start from each pattern with ``noise`` of its bits flipped, run asynchronous
    updates, and record the overlap with the intended pattern.

    Returns ``(alphas, mean_overlap, frac_recalled)`` where a pattern counts as
    recalled when its final overlap ≥ 0.95.
    """
    if alphas is None:
        alphas = np.round(np.arange(0.02, 0.281, 0.01), 3)
    alphas = np.asarray(alphas, dtype=float)
    rng = np.random.default_rng(seed)

    mean_m = np.zeros(len(alphas))
    frac = np.zeros(len(alphas))
    for ai, a in enumerate(alphas):
        p = max(1, int(round(a * n)))
        ms = []
        for _ in range(trials):
            pats = rng.choice([-1.0, 1.0], size=(p, n))
            W = hebbian_weights(pats)
            mu = rng.integers(p)
            start = corrupt(pats[mu], noise, rng)
            final, _ = async_update(W, start, rng, sweeps=sweeps)
            ms.append(abs(overlap(final, pats[mu])))
        ms = np.asarray(ms)
        mean_m[ai] = ms.mean()
        frac[ai] = float((ms >= 0.95).mean())
    return alphas, mean_m, frac


def critical_alpha(alphas: np.ndarray, frac: np.ndarray, level: float = 0.5) -> float:
    """Linearly interpolate the α where the recall fraction crosses ``level``."""
    alphas = np.asarray(alphas, dtype=float)
    frac = np.asarray(frac, dtype=float)
    for i in range(len(alphas) - 1):
        if frac[i] >= level > frac[i + 1]:
            t = (frac[i] - level) / (frac[i] - frac[i + 1])
            return float(alphas[i] + t * (alphas[i + 1] - alphas[i]))
    return float("nan")


# ──────────────────────────────────────────────────────── modern Hopfield ──


def modern_update(patterns: np.ndarray, query: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """One step of the continuous Hopfield update of Ramsauer et al. (2020):

        ξ_new = Xᵀ · softmax(β · X · ξ)

    with ``patterns`` = X of shape (P, N) and ``query`` = ξ of shape (N,).
    Read it as attention: ξ is the query, the stored patterns are both keys and
    values, β is 1/√d, and the softmax weights are the attention row. Large β
    puts all the mass on one pattern (exact recall in a single step); small β
    averages patterns together (a blurred metastable state).

    Returns ``(retrieved, weights)`` — the new continuous state and the softmax
    row that produced it, so you can look at the attention distribution itself.
    """
    X = np.asarray(patterns, dtype=float)
    q = np.asarray(query, dtype=float)
    logits = beta * (X @ q)
    w = np.exp(logits - logits.max())
    w /= w.sum()
    return X.T @ w, w


# ───────────────────────────────────────────────────────────────  glyphs ──

GLYPHS: dict[str, list[str]] = {
    "A": [
        "..######..",
        ".##....##.",
        "##......##",
        "##......##",
        "##########",
        "##......##",
        "##......##",
        "##......##",
        "##......##",
        "##......##",
    ],
    "T": [
        "##########",
        "##########",
        "....##....",
        "....##....",
        "....##....",
        "....##....",
        "....##....",
        "....##....",
        "....##....",
        "....##....",
    ],
    "Z": [
        "##########",
        "##########",
        ".......##.",
        "......##..",
        ".....##...",
        "....##....",
        "...##.....",
        "..##......",
        "##########",
        "##########",
    ],
    "L": [
        "##........",
        "##........",
        "##........",
        "##........",
        "##........",
        "##........",
        "##........",
        "##........",
        "##########",
        "##########",
    ],
    "X": [
        "##......##",
        "###....###",
        ".###..###.",
        "..######..",
        "...####...",
        "...####...",
        "..######..",
        ".###..###.",
        "###....###",
        "##......##",
    ],
}

GLYPH_SHAPE = (10, 10)


def glyph_patterns(letters: str = "ATZLX") -> tuple[np.ndarray, list[str]]:
    """Return ``(P, N)`` ±1 patterns for the requested letters plus their names."""
    names = list(letters)
    rows = []
    for name in names:
        art = GLYPHS[name]
        flat = "".join(art)
        rows.append([1.0 if ch == "#" else -1.0 for ch in flat])
    return np.asarray(rows, dtype=float), names


def as_image(vec: np.ndarray) -> np.ndarray:
    """Reshape a flat ±1 state back to the 10×10 grid for imshow."""
    return np.asarray(vec, dtype=float).reshape(GLYPH_SHAPE)
