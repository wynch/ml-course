"""Masked diffusion and block diffusion on characters: losses, scores, samplers.

Forward process. At time t in (0, 1], each character is replaced by [MASK]
with probability t, independently. At t = 1 the line is all [MASK]; at t = 0
it is the data. The schedule is linear, as in LLaDA and MDLM.

Training. The denoiser sees the masked line and predicts every masked
character. The loss is the cross-entropy on the masked positions, weighted
by 1/t and divided by the line length:

    L = E_t E_mask [ (1/t) * sum_{i masked} -log p(x_i | x_t) ] / T

Its expectation is an upper bound on the line's negative log-likelihood per
character (the ELBO, in nats), so it can be set beside an autoregressive
model's exact loss; it is biased against the diffusion model.

Block diffusion (Arriola et al., 2025) cuts the line into blocks of L
characters, runs the same process inside each block, and goes left to right
across blocks: a block sees the clean blocks before it and nothing after.
L = 1 is an autoregressive model; L = T is plain masked diffusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS_T = 1e-3          # smallest masking rate drawn in training, as LLaDA's eps


# ------------------------------------------------------------------ forward process
def mask_line(x0: torch.Tensor, t: torch.Tensor, mask_id: int, gen=None) -> tuple[torch.Tensor, torch.Tensor]:
    """x0 (B, T); t (B,) or (B, T) masking rates. Returns (x_t, masked bool)."""
    if t.dim() == 1:
        t = t[:, None]
    u = torch.rand(x0.shape, generator=gen, device="cpu").to(x0.device)
    masked = u < t
    return torch.where(masked, torch.full_like(x0, mask_id), x0), masked


def stratified_t(n: int, gen=None, device="cpu") -> torch.Tensor:
    """n masking rates that cover (EPS_T, 1] evenly: one uniform shift, then n equal strata.
    Lower-variance than n independent draws; each one is still uniform."""
    u = torch.rand(1, generator=gen).item()
    t = (u + torch.arange(n, dtype=torch.float32)) / n
    return (EPS_T + (1 - EPS_T) * t).to(device)


# ------------------------------------------------------------------ losses
def ar_loss(model, x0: torch.Tensor) -> torch.Tensor:
    """Next-character cross-entropy: positions 0..T-2 predict 1..T-1."""
    logits = model(x0[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x0[:, 1:].reshape(-1))


def mdm_loss(model, x0: torch.Tensor, t: torch.Tensor, gen=None) -> torch.Tensor:
    """The 1/t-weighted masked cross-entropy, per character, mean over the batch."""
    xt, masked = mask_line(x0, t, model.cfg.mask_id, gen)
    logits = model(xt)
    ce = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")       # (B, T)
    return ((ce * masked) / t[:, None]).sum(1).div(x0.size(1)).mean()


def block_mask(T: int, L: int, device="cpu") -> torch.Tensor:
    """Attention pattern over [noisy line ; clean line], (2T, 2T), True = may attend.

    noisy block b -> noisy block b (both ways) and clean blocks before b
    clean block b -> clean blocks up to and including b (both ways inside b)
    clean         -> never the noisy half
    """
    b = torch.arange(T, device=device) // L
    same = b[:, None] == b[None, :]
    before = b[None, :] < b[:, None]
    upto = b[None, :] <= b[:, None]
    top = torch.cat([same, before], 1)
    bottom = torch.cat([torch.zeros_like(same), upto], 1)
    return torch.cat([top, bottom], 0)


def bd_forward(model, xt: torch.Tensor, x0: torch.Tensor, L: int) -> torch.Tensor:
    """Logits for the noisy half, computed in one pass over [x_t ; x_0] with block_mask."""
    T = x0.size(1)
    pos = torch.arange(T, device=x0.device).repeat(2)
    logits = model(torch.cat([xt, x0], 1), attn_mask=block_mask(T, L, x0.device), pos=pos)
    return logits[:, :T]


def bd_loss(model, x0: torch.Tensor, L: int, gen=None) -> torch.Tensor:
    """Block diffusion loss: every block draws its own masking rate."""
    B, T = x0.shape
    nb = T // L
    t_blocks = torch.stack([stratified_t(B, gen) for _ in range(nb)], 1)              # (B, nb)
    t = t_blocks.repeat_interleave(L, 1).to(x0.device)                       # (B, T)
    xt, masked = mask_line(x0, t, model.cfg.mask_id, gen)
    ce = F.cross_entropy(bd_forward(model, xt, x0, L).transpose(1, 2), x0, reduction="none")
    return ((ce * masked) / t).sum(1).div(T).mean()


# ------------------------------------------------------------------ scores on held-out text
@torch.no_grad()
def ar_nll(model, blocks: torch.Tensor, log_unigram: torch.Tensor, batch: int = 64) -> float:
    """Exact NLL per character of each whole block, nats. Training windows start anywhere, so
    the model never predicts a window's first character; it gets the training split's
    character frequencies (log_unigram), and the other T - 1 come from the model."""
    model.eval()
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    for i in range(0, len(blocks), batch):
        x = blocks[i: i + batch].to(dev)
        logits = model(x[:, :-1])
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum").item()
        tot -= log_unigram[x[:, 0].cpu()].sum().item()
        n += x.numel()
    return tot / n


@torch.no_grad()
def elbo(model, blocks: torch.Tensor, K: int = 32, L: int | None = None, seed: int = 0, batch: int = 64) -> float:
    """The diffusion bound per character, nats: K masking rates at the midpoints of K equal strata
    of (0, 1], one random mask per block and rate. L given: block diffusion (rates shared by
    all blocks of a line; any schedule gives the same expectation)."""
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    tot, n = 0.0, 0
    for k in range(K):
        tk = (k + 0.5) / K
        for i in range(0, len(blocks), batch):
            x0 = blocks[i: i + batch].to(dev)
            t = torch.full((len(x0),), tk, device=dev)
            xt, masked = mask_line(x0, t, model.cfg.mask_id, gen)
            logits = model(xt) if L is None else bd_forward(model, xt, x0, L)
            ce = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")
            tot += ((ce * masked).sum() / tk).item()
            n += x0.numel()
    return tot / n


@torch.no_grad()
def ordered_nll(model, blocks: torch.Tensor, order: str = "l2r", seed: int = 0) -> float:
    """Exact NLL per character of a masked diffusion model that reveals one character at a time
    in a fixed order: the product of its conditionals is a proper distribution over lines.
    order: 'l2r', 'r2l' or 'random' (one random order per block)."""
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    M = model.cfg.mask_id
    tot, n = 0.0, 0
    for x0 in blocks:
        T = len(x0)
        if order == "l2r":
            perm = torch.arange(T)
        elif order == "r2l":
            perm = torch.arange(T - 1, -1, -1)
        else:
            perm = torch.randperm(T, generator=gen)
        rank = torch.empty(T, dtype=torch.long)
        rank[perm] = torch.arange(T)
        # row s: the characters revealed before step s, the rest masked; it scores perm[s]
        seen = rank[None, :] < torch.arange(T)[:, None]
        xin = torch.where(seen, x0[None, :], torch.full((T, T), M)).to(dev)
        logits = model(xin)
        lp = F.log_softmax(logits[torch.arange(T), perm.to(dev)], -1)
        tot -= lp[torch.arange(T), x0[perm].to(dev)].sum().item()
        n += T
    return tot / n


# ------------------------------------------------------------------ sampling
def gumbel_argmax(logits: torch.Tensor, temperature: float, gen=None) -> torch.Tensor:
    """Sample from softmax(logits / temperature) by the Gumbel-max trick, in float64 as LLaDA
    does (low precision here hurts samples; Zheng et al., 2024). temperature 0 = argmax."""
    if temperature == 0:
        return logits.argmax(-1)
    u = torch.rand(logits.shape, generator=gen, dtype=torch.float64)       # MPS has no float64: CPU
    noisy = logits.detach().cpu().double().exp() / (-torch.log(u)) ** temperature
    return noisy.argmax(-1).to(logits.device)


def reveal_counts(n_masked: int, steps: int) -> list[int]:
    """How many characters each step reveals: equal shares, the remainder on the first steps."""
    base, rem = divmod(n_masked, steps)
    return [base + (1 if i < rem else 0) for i in range(steps)]


def choose(logits, x, masked, k, rule, temperature, gen):
    """One denoising step on a batch: guess every masked character, keep k per line, by the rule
    'confidence' (the k guesses the model gives the highest probability) or 'random'."""
    guess = gumbel_argmax(logits, temperature, gen)
    if rule == "confidence":
        p = F.softmax(logits.float(), -1)
        score = p.gather(-1, guess[..., None]).squeeze(-1)
    else:
        score = torch.rand(guess.shape, generator=gen).to(guess.device)
    score = torch.where(masked, score, torch.full_like(score, -math.inf))
    keep = torch.zeros_like(masked)
    keep.scatter_(1, score.topk(k, dim=1).indices, True)
    return torch.where(keep, guess, x), keep


@torch.no_grad()
def sample_mdm(model, n: int, steps: int, rule: str = "confidence", temperature: float = 1.0, T: int = 128,
               x_init: torch.Tensor | None = None, gen=None, trace: bool = False):
    """Start from an all-[MASK] line, or from x_init (n, T) whose [MASK] positions are to be
    filled, and reveal the masked characters in `steps` steps. Returns ids (n, T), and with
    trace the step (1-based) at which each position was revealed (0 = given)."""
    model.eval()
    dev = next(model.parameters()).device
    M = model.cfg.mask_id
    x = torch.full((n, T), M, dtype=torch.long, device=dev) if x_init is None else x_init.clone().to(dev)
    when = torch.zeros(x.shape, dtype=torch.long, device=dev)
    counts = reveal_counts(int((x[0] == M).sum()), steps)
    for s, k in enumerate(counts):
        if k == 0:
            continue
        masked = x == M
        x, keep = choose(model(x), x, masked, k, rule, temperature, gen)
        when[keep] = s + 1
    return (x, when) if trace else x


@torch.no_grad()
def continue_ar(model, prefix: torch.Tensor, n_new: int, temperature: float = 0.0, gen=None) -> torch.Tensor:
    """Extend each prefix (n, P) by n_new characters, left to right; returns only the new ones."""
    model.eval()
    dev = next(model.parameters()).device
    x = prefix.clone().to(dev)
    for _ in range(n_new):
        x = torch.cat([x, gumbel_argmax(model(x)[:, -1], temperature, gen)[:, None]], 1)
    return x[:, prefix.size(1):]


@torch.no_grad()
def sample_ar(model, n: int, log_unigram: torch.Tensor, T: int = 128, temperature: float = 1.0, gen=None,
              cache: bool = True) -> torch.Tensor:
    """Left to right, one character per forward pass; the first character from the training
    split's frequencies, as ar_nll scores it. cache=False recomputes the whole prefix each step."""
    model.eval()
    dev = next(model.parameters()).device
    x = gumbel_argmax(log_unigram.expand(n, -1), temperature, gen)[:, None].to(dev)
    past, last = None, x
    for i in range(T - 1):
        if cache:
            logits, kvs = model(last, pos=torch.tensor([i], device=dev), past=past, return_kv=True)
            past = _extend(past, kvs)
            logits = logits[:, -1]
        else:
            logits = model(x)[:, -1]
        last = gumbel_argmax(logits, temperature, gen)[:, None]
        x = torch.cat([x, last], 1)
    return x


def _extend(cache, kvs):
    from model import extend_cache
    return extend_cache(cache, kvs)


@torch.no_grad()
def sample_bd(model, n: int, L: int, steps_per_block: int, rule: str = "confidence", temperature: float = 1.0,
              T: int = 128, gen=None, cache: bool = True, trace: bool = False):
    """Block diffusion sampling: blocks of L characters left to right, each denoised from all
    [MASK] in steps_per_block steps while it attends to the clean blocks before it.

    cache=True keeps the keys and values of finished blocks and runs the network on the current
    block only (plus one pass per finished block to store its keys and values).
    cache=False runs the network on [clean prefix ; current block] at every step, with the
    attention pattern of training. The two give the same characters up to float rounding."""
    model.eval()
    dev = next(model.parameters()).device
    M = model.cfg.mask_id
    out = torch.zeros((n, 0), dtype=torch.long, device=dev)
    when = torch.zeros((n, 0), dtype=torch.long, device=dev)
    past = None
    for b in range(T // L):
        pos = torch.arange(b * L, (b + 1) * L, device=dev)
        x = torch.full((n, L), M, dtype=torch.long, device=dev)
        w = torch.zeros((n, L), dtype=torch.long, device=dev)
        for s, k in enumerate(reveal_counts(L, steps_per_block)):
            if k == 0:
                continue
            if cache:
                logits = model(x, pos=pos, past=past)
            else:
                P = out.size(1)
                allpos = torch.arange(P + L, device=dev)
                m = torch.ones(P + L, P + L, dtype=torch.bool, device=dev)
                if P:
                    bl = torch.arange(P, device=dev) // L
                    m[:P, :P] = bl[None, :] <= bl[:, None]       # clean prefix: block-causal
                    m[:P, P:] = False
                logits = model(torch.cat([out, x], 1), attn_mask=m, pos=allpos)[:, P:]
            x, keep = choose(logits, x, x == M, k, rule, temperature, gen)
            w[keep] = b * steps_per_block + s + 1
        if cache:
            _, kvs = model(x, pos=pos, past=past, return_kv=True)
            past = _extend(past, kvs)
        out = torch.cat([out, x], 1)
        when = torch.cat([when, w], 1)
    return (out, when) if trace else out


def network_calls(kind: str, steps: int, L: int = 128, T: int = 128, cache: bool = True) -> dict:
    """Forward passes and positions processed per line, for the cost table."""
    if kind == "ar":
        return {"calls": T - 1, "positions": T - 1 if cache else T * (T - 1) // 2}
    if kind == "mdm":
        return {"calls": steps, "positions": steps * T}
    nb = T // L
    per_block = steps // nb
    if cache:
        return {"calls": nb * per_block + nb, "positions": nb * per_block * L + T}
    return {"calls": nb * per_block, "positions": sum(per_block * (b + 1) * L for b in range(nb))}
