"""Small sequence models for multi-query associative recall (MQAR).

Every model is the same shell: an embedding, L blocks, a norm and a head.
A block is pre-norm and residual: x + mixer(norm(x)), then x + mlp(norm(x)).
Only the mixer differs:

- ``MemoryMixer(rule)``: one of the four write rules of ``memory.py``
  (linear, decay, delta, gated_delta), H heads of size d_k = d_v.
- ``AttentionMixer``: causal softmax attention, the same projections.

Both mixers pass q, k and v through a short causal convolution (kernel 4)
first, as DeltaNet and Zoology's models do. With it, the token before a value
can shape the key written at the value's position, so one layer can bind a
key to the value that follows it. No model has position embeddings; the
convolution is the only sense of order the attention model has, and the
causal mask.

State between tokens, per layer: H * d_k * d_v floats for a memory mixer;
2 * T * H * d_k for attention's KV cache, growing with T.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import memory


class ShortConv(nn.Module):
    """Causal depthwise convolution over time, kernel 4, then SiLU."""

    def __init__(self, ch: int, kernel: int = 4):
        super().__init__()
        self.kernel = kernel
        self.weight = nn.Parameter(torch.randn(kernel, ch) / math.sqrt(kernel))   # weight[j]: lag j
        self.bias = nn.Parameter(torch.zeros(ch))

    def forward(self, x):                                     # (B, T, ch)
        # y_t = sum_j w_j * x_{t-j}, written as shifts: on the CPU this runs far
        # faster than a grouped Conv1d, whose backward pass is slow there.
        y = x * self.weight[0] + self.bias
        for j in range(1, self.kernel):
            y = y + F.pad(x, (0, 0, j, 0))[:, : x.shape[1]] * self.weight[j]
        return F.silu(y)


class _Mixer(nn.Module):
    def __init__(self, d: int, d_head: int):
        super().__init__()
        assert d % d_head == 0
        self.H, self.dk = d // d_head, d_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = ShortConv(3 * d)
        self.out = nn.Linear(d, d, bias=False)

    def _heads(self, x):
        B, T, _ = x.shape
        q, k, v = self.conv(self.qkv(x)).chunk(3, dim=-1)
        return [t.view(B, T, self.H, self.dk).transpose(1, 2) for t in (q, k, v)]   # (B, H, T, dk)

    def _merge(self, o):
        B, H, T, dv = o.shape
        return self.out(o.transpose(1, 2).reshape(B, T, H * dv))


BETA_BIAS = -2.0      # write strengths start near sigmoid(-2) = 0.12; see README section 3
ALPHA_BIAS = 4.6      # decays start near sigmoid(4.6) = 0.99; at 0.95 the gated delta rule never learned (README section 3)


class MemoryMixer(_Mixer):
    """A matrix state per head, written by one of the four rules."""

    def __init__(self, d: int, d_head: int, rule: str, chunk: int = 64, alpha_bias: float = ALPHA_BIAS):
        super().__init__(d, d_head)
        self.rule, self.chunk = rule, chunk
        self.delta, self.gated = memory.RULES[rule]
        self.beta = nn.Linear(d, self.H)                      # write strength, sigmoid
        nn.init.constant_(self.beta.bias, BETA_BIAS)
        if self.gated:
            self.alpha = nn.Linear(d, self.H)                 # decay, log-sigmoid
            nn.init.constant_(self.alpha.bias, alpha_bias)
        self.onorm = nn.RMSNorm(d_head)

    def gates(self, x):
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)    # (B, H, T)
        if self.gated:
            log_alpha = F.logsigmoid(self.alpha(x)).transpose(1, 2)
        else:
            log_alpha = torch.zeros_like(beta)
        return beta, log_alpha

    def forward(self, x, recurrent: bool = False):
        q, k, v = self._heads(x)
        q, k = memory.unit(q), memory.unit(k)                 # unit keys keep the delta rule stable
        beta, log_alpha = self.gates(x)
        if recurrent:
            o, _ = memory.recurrent(q, k, v, beta, log_alpha, self.delta)
        else:
            o, _ = memory.chunked(q, k, v, beta, log_alpha, self.delta, chunk=self.chunk)
        return self._merge(self.onorm(o))

    def state_floats(self) -> int:
        return self.H * self.dk * self.dk


class AttentionMixer(_Mixer):
    """Causal softmax attention: the ceiling. Its cache grows with T."""

    def forward(self, x, recurrent: bool = False):
        q, k, v = self._heads(x)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self._merge(o)

    def state_floats(self, T: int) -> int:
        return 2 * T * self.H * self.dk


class Block(nn.Module):
    def __init__(self, d: int, mixer: nn.Module):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.mixer = mixer
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, recurrent=False):
        x = x + self.mixer(self.n1(x), recurrent=recurrent)
        return x + self.mlp(self.n2(x))


class RecallLM(nn.Module):
    """``layers`` is a list of mixer kinds: "attention" or a rule name."""

    def __init__(self, vocab: int, d: int, d_head: int, layers: list[str], chunk: int = 64,
                 alpha_bias: float = None):
        super().__init__()
        self.layers = list(layers)
        self.emb = nn.Embedding(vocab, d)
        mix = lambda kind: (AttentionMixer(d, d_head) if kind == "attention"
                            else MemoryMixer(d, d_head, kind, chunk, ALPHA_BIAS if alpha_bias is None else alpha_bias))
        self.blocks = nn.ModuleList([Block(d, mix(kind)) for kind in layers])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx, recurrent: bool = False):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, recurrent=recurrent)
        return self.head(self.norm(x))

    def state_floats(self, T: int) -> int:
        """Floats a model must keep between tokens after T tokens (convolution buffers left out)."""
        n = 0
        for b in self.blocks:
            m = b.mixer
            n += m.state_floats(T) if isinstance(m, AttentionMixer) else m.state_floats()
        return n


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ MQAR data
class MQAR:
    """Multi-query associative recall (Arora et al., 2023, arXiv 2312.04927), our variant.

    Vocabulary of ``vocab`` tokens: keys are 1 .. vocab/2 - 1, values vocab/2 .. vocab - 1.
    A sequence of length T holds n_pairs (key, value) pairs, keys all different,
    then T - 2 n_pairs queries: each a key drawn from the pairs, whose target is
    its value. Loss and accuracy count the query positions only.
    """

    def __init__(self, vocab: int = 512, T: int = 256):
        self.vocab, self.T, self.half = vocab, T, vocab // 2

    def batch(self, n_pairs: int, batch: int, gen: torch.Generator):
        T, half = self.T, self.half
        assert 2 * n_pairs < T and n_pairs < half
        keys = torch.argsort(torch.rand(batch, half - 1, generator=gen), dim=1)[:, :n_pairs] + 1
        vals = torch.randint(half, self.vocab, (batch, n_pairs), generator=gen)
        x = torch.zeros(batch, T, dtype=torch.long)
        y = torch.full((batch, T), -100, dtype=torch.long)    # -100: not scored
        x[:, 0 : 2 * n_pairs : 2], x[:, 1 : 2 * n_pairs : 2] = keys, vals
        nq = T - 2 * n_pairs
        pick = torch.randint(0, n_pairs, (batch, nq), generator=gen)
        x[:, 2 * n_pairs :] = keys.gather(1, pick)
        y[:, 2 * n_pairs :] = vals.gather(1, pick)
        return x, y


def mqar_loss(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)


@torch.no_grad()
def mqar_accuracy(model, x, y) -> float:
    pred = model(x).argmax(-1)
    m = y != -100
    return (pred[m] == y[m]).float().mean().item()


def lr_at(step: int, steps: int, peak: float, warm: int = 100) -> float:
    if step < warm:
        return peak * (step + 1) / warm
    p = (step - warm) / max(1, steps - warm)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
