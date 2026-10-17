"""The three sequence models of this module, each with a parallel forward for
training and a one-token ``step`` for generation.

- ``GRULM``: a gated recurrent network (section 1).
- ``SSMLM``: a stack of Mamba-style blocks. ``selective=False`` gives a diagonal
  linear recurrence whose decays and input weights are the same at every step
  (section 2, first half); ``selective=True`` computes them from each input
  (section 2, second half).
- ``load_gpt`` / ``GPTCache``: module 04's GPT, rebuilt from its own model.py,
  with a KV cache added for one-token steps (section 3).

Every model maps (batch, T) char ids to (batch, T, vocab) logits. Every
``step`` maps (batch,) ids and a state to (batch, vocab) logits and the new
state. ``state_floats(state)`` counts what a model must keep between tokens.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import M04


# ------------------------------------------------------------------ GRU
class GRULM(nn.Module):
    """Embedding -> stacked GRU -> LayerNorm -> head tied to the embedding.

    The GRU update, per layer and step (nn.GRU's convention):
        r = sigmoid(W_ir x + W_hr h + b_r)          reset gate
        z = sigmoid(W_iz x + W_hz h + b_z)          update gate
        n = tanh(W_in x + r * (W_hn h + b_hn))      candidate
        h = (1 - z) * n + z * h
    """

    def __init__(self, vocab_size: int, hidden: int = 224, n_layer: int = 2):
        super().__init__()
        self.hidden, self.n_layer = hidden, n_layer
        self.emb = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, n_layer, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx, targets=None):
        h, _ = self.rnn(self.emb(idx))
        logits = self.head(self.ln(h))
        return logits, _loss(logits, targets)

    def init_state(self, batch: int, device):
        return torch.zeros(self.n_layer, batch, self.hidden, device=device)

    @torch.no_grad()
    def step(self, idx_t, state):
        out, state = self.rnn(self.emb(idx_t)[:, None, :], state)
        return self.head(self.ln(out[:, 0])), state

    @staticmethod
    def state_floats(state) -> int:
        return state.numel()


# ------------------------------------------------------------------ SSM
class _Block(nn.Module):
    """The shell both SSM blocks share, pre-norm and residual:

        u     = LayerNorm(x)
        x', z = in_proj(u)                     two streams of width E
        x'    = silu(causal depthwise conv(x'), kernel 4)
        y     = core(x')                       the recurrence: the only part that differs
        out   = x + out_proj((y + D * x') * silu(z))
    """

    def __init__(self, d: int, E: int, extra: int, d_conv: int = 4):
        super().__init__()
        self.E, self.d_conv = E, d_conv
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * E + extra, bias=False)
        self.conv = nn.Conv1d(E, E, d_conv, groups=E, padding=d_conv - 1)
        self.D = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, d, bias=False)

    def _split(self, p):
        return p[..., : self.E], p[..., self.E : 2 * self.E], p[..., 2 * self.E :]

    def forward(self, x):
        T = x.shape[1]
        xs, z, extra = self._split(self.in_proj(self.norm(x)))
        xs = F.silu(self.conv(xs.transpose(1, 2))[..., :T].transpose(1, 2))
        y = self.core(xs, extra)
        return x + self.out_proj((y + self.D * xs) * F.silu(z))

    def init_state(self, batch: int, device):
        return (torch.zeros(batch, self.E, self.d_conv - 1, device=device),   # last conv inputs
                torch.zeros(batch, *self.state_shape, device=device))         # h

    def step(self, x_t, state):
        conv_buf, h = state
        xs, z, extra = self._split(self.in_proj(self.norm(x_t)))
        window = torch.cat([conv_buf, xs[:, :, None]], dim=-1)                # (B, E, d_conv)
        xs = F.silu((window * self.conv.weight[:, 0]).sum(-1) + self.conv.bias)
        y, h = self.core_step(xs, extra, h)
        return x_t + self.out_proj((y + self.D * xs) * F.silu(z)), (window[:, :, 1:], h)


def _dt_init(n: int) -> torch.Tensor:
    """softplus^-1 of step sizes drawn log-uniform in [1e-3, 1e-1], as in Mamba."""
    dt = torch.exp(torch.rand(n) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
    return dt + torch.log(-torch.expm1(-dt))


class LTIBlock(_Block):
    """Diagonal linear recurrence, the same at every step (S4D-style).

    Each of the E channels carries N states with their own decay:
        h_t = exp(dt * A) * h_{t-1} + dt * B * x_t        A < 0, all (E, N)
        y_t = sum_n C * h_t
    Nothing here depends on the input, so unrolling the recurrence gives a fixed
    kernel, K[e, k] = sum_n C B dt exp(k dt A), and y is the causal convolution
    of x with K. Training uses the convolution; generation uses the recurrence.
    """

    def __init__(self, d: int, expand: int = 2, n_state: int = 16):
        E = expand * d
        super().__init__(d, E, extra=0)
        self.N = n_state
        self.state_shape = (E, n_state)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_state + 1).float()).repeat(E, 1))
        self.B = nn.Parameter(torch.randn(E, n_state) / math.sqrt(n_state))
        self.C = nn.Parameter(torch.randn(E, n_state) / math.sqrt(n_state))
        self.dt_bias = nn.Parameter(_dt_init(E))

    def _decay(self):
        dt = F.softplus(self.dt_bias)                                         # (E,)
        return dt, torch.exp(dt[:, None] * -torch.exp(self.A_log))           # (E, N) in (0, 1)

    def kernel(self, T: int) -> torch.Tensor:
        dt, a = self._decay()
        k = torch.arange(T, device=a.device, dtype=a.dtype)
        powers = torch.exp(k[None, None, :] * torch.log(a)[..., None])      # a^k, (E, N, T)
        return ((self.C * self.B * dt[:, None])[..., None] * powers).sum(1)   # (E, T)

    def core(self, xs, extra):
        T = xs.shape[1]
        K = self.kernel(T)
        idx = torch.arange(T, device=xs.device)
        lag = idx[:, None] - idx[None, :]                                     # t - s
        toeplitz = K[:, lag.clamp(min=0)] * (lag >= 0)                      # (E, t, s)
        return (toeplitz @ xs.permute(2, 1, 0)).permute(2, 1, 0)              # (E, t, B) -> (B, t, E)

    def core_step(self, xs, extra, h):
        dt, a = self._decay()
        h = a * h + (dt * xs)[..., None] * self.B
        return (h * self.C).sum(-1), h


class SelectiveBlock(_Block):
    """Selective recurrence: step size, B and C are computed from each input.

    Mamba-2's form (Dao and Gu, 2024). The E channels split into H heads of
    P channels. Per head h and step t, one scalar decay:
        a_t   = exp(dt_t A_h)                            dt_t = softplus(.) from the input
        S_t   = a_t * S_{t-1} + dt_t * x_t B_t^T          S is (P, N) per head
        y_t   = S_t C_t                                   B_t, C_t (N,) from the input
    Unrolled, y_t = sum_{s<=t} (C_t . B_s) * exp(sum_{r=s+1..t} log a_r) * dt_s x_s:
    a (T, T) matrix, causal and decaying, applied to the inputs. Training uses
    that matrix form; generation uses the recurrence. (Mamba-1 gives every
    channel and state its own decay, which rules out the matrix form; in plain
    PyTorch it needs a loop over t, and a step of this module took seconds.)
    """

    def __init__(self, d: int, expand: int = 2, n_state: int = 16, n_head: int = 8):
        E = expand * d
        H = n_head
        assert E % H == 0
        super().__init__(d, E, extra=2 * n_state + H)
        self.N, self.H, self.P = n_state, H, E // H
        self.state_shape = (H, E // H, n_state)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, H)))
        self.dt_bias = nn.Parameter(_dt_init(H))

    def _params(self, extra):
        Bm, Cm, dt = extra[..., : self.N], extra[..., self.N : 2 * self.N], extra[..., 2 * self.N :]
        dt = F.softplus(dt + self.dt_bias)                                    # (..., H)
        return Bm, Cm, dt, dt * -torch.exp(self.A_log)                        # last: log a_t

    def core(self, xs, extra):
        Bsz, T, _ = xs.shape
        Bm, Cm, dt, log_a = self._params(extra)
        L = torch.cumsum(log_a, dim=1).transpose(1, 2)                        # (B, H, T)
        seg = L[..., :, None] - L[..., None, :]                               # (B, H, t, s): sum of log a over s+1..t
        causal = torch.ones(T, T, dtype=torch.bool, device=xs.device).tril()
        decay = torch.exp(seg.masked_fill(~causal, -torch.inf))
        M = (Cm @ Bm.transpose(1, 2))[:, None] * decay                        # (B, H, t, s)
        u = (xs.view(Bsz, T, self.H, self.P) * dt[..., None]).transpose(1, 2)  # (B, H, s, P)
        return (M @ u).transpose(1, 2).reshape(Bsz, T, self.E)

    def core_step(self, xs, extra, S):
        Bm, Cm, dt, log_a = self._params(extra)                               # (B, N), (B, N), (B, H)
        x = xs.view(-1, self.H, self.P)
        S = torch.exp(log_a)[..., None, None] * S + (dt[..., None] * x)[..., None] * Bm[:, None, None, :]
        return (S * Cm[:, None, None, :]).sum(-1).reshape(-1, self.E), S


class SSMLM(nn.Module):
    def __init__(self, vocab_size: int, selective: bool = True, d: int = 128, n_layer: int = 5,
                 n_state: int = 16):
        super().__init__()
        Block = SelectiveBlock if selective else LTIBlock
        self.emb = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList([Block(d, n_state=n_state) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        return logits, _loss(logits, targets)

    def init_state(self, batch: int, device):
        return [b.init_state(batch, device) for b in self.blocks]

    @torch.no_grad()
    def step(self, idx_t, state):
        x = self.emb(idx_t)
        new = []
        for b, s in zip(self.blocks, state):
            x, s = b.step(x, s)
            new.append(s)
        return self.head(self.ln(x)), new

    @staticmethod
    def state_floats(state) -> int:
        return sum(c.numel() + h.numel() for c, h in state)


# ------------------------------------------------------------------ GPT
def _m04_model_module():
    # Loaded by path under its own name, so it never shadows another "model".
    spec = importlib.util.spec_from_file_location("m04_model", M04 / "python" / "src" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m04_model"] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


def make_gpt(vocab_size: int, block_size: int = 128, dropout: float = 0.0):
    """Module 04's GPT class with module 04's sizes: 3 layers, 4 heads, d_model 128."""
    m04 = _m04_model_module()
    cfg = m04.GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=3, n_head=4,
                        d_model=128, dropout=dropout)
    return m04.GPT(cfg)


def load_committed_gpt():
    """Module 04's trained weights, from its committed flat float32 export."""
    import json

    import numpy as np

    art = M04 / "artifacts"
    cfg = json.loads((art / "tiny_gpt_config.json").read_text())
    model = make_gpt(cfg["vocab_size"], cfg["block_size"])
    blob = np.fromfile(art / "tiny_gpt_weights.bin", dtype="<f4")
    state = {}
    for t in cfg["tensors"]:
        s = t["offset"] // 4
        state[t["name"]] = torch.from_numpy(blob[s : s + t["count"]].reshape(t["shape"]).copy())
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k == "head.weight" or k.endswith("attn.mask") for k in missing)
    return model.eval()


class GPTCache:
    """One-token steps for module 04's GPT, with a KV cache.

    Module 04's own ``generate`` reruns the whole context for every new token.
    Here each step computes q, k, v for the new token only, appends k and v to
    the cache, and attends over the cache. Module 07 does the same in Zig.
    """

    def __init__(self, gpt):
        self.gpt = gpt

    def init_state(self, batch: int, device):
        return {"pos": 0, "k": [None] * len(self.gpt.blocks), "v": [None] * len(self.gpt.blocks)}

    @torch.no_grad()
    def step(self, idx_t, state):
        g = self.gpt
        pos = state["pos"]
        x = g.wte(idx_t) + g.wpe.weight[pos]                                    # (B, C)
        for i, blk in enumerate(g.blocks):
            a = blk.attn
            q, k, v = a.qkv(blk.ln1(x)).split(a.d_model, dim=-1)
            B = x.shape[0]
            q, k, v = (t.view(B, a.n_head, 1, a.d_head) for t in (q, k, v))
            K = k if state["k"][i] is None else torch.cat([state["k"][i], k], dim=2)
            V = v if state["v"][i] is None else torch.cat([state["v"][i], v], dim=2)
            state["k"][i], state["v"][i] = K, V
            att = torch.softmax((q @ K.transpose(-2, -1)) / math.sqrt(a.d_head), dim=-1)
            x = x + a.proj((att @ V).reshape(B, a.d_model))
            x = x + blk.mlp(blk.ln2(x))
        state["pos"] = pos + 1
        return g.head(g.ln_f(x)), state

    @staticmethod
    def state_floats(state) -> int:
        return sum(t.numel() for t in state["k"] + state["v"] if t is not None)


# ------------------------------------------------------------------ shared
def _loss(logits, targets):
    if targets is None:
        return None
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def build(kind: str, vocab_size: int):
    """The four models at the tiny GPT's size (619,776 parameters), within 3%."""
    if kind == "gpt":
        return make_gpt(vocab_size)
    if kind == "gru":
        return GRULM(vocab_size, hidden=224, n_layer=2)
    if kind == "lti":
        return SSMLM(vocab_size, selective=False, d=132, n_layer=5)
    if kind == "selective":
        return SSMLM(vocab_size, selective=True, d=140, n_layer=5)
    raise ValueError(kind)


def n_params(model: nn.Module) -> int:
    """Unique parameters: a tied head counts once."""
    return sum(p.numel() for p in model.parameters())
