"""Batched sampling, rewards and log-probabilities for the char-GPT.

Every prompt has exactly PROMPT_LEN characters, so a batch needs no left padding. A
reply runs until its first END or for MAX_REPLY characters; the mask marks
the characters up to and including that END. Characters after it are PAD
and never count.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import task
from gpt import GPT, GPTConfig


# Smaller than module 04's 3 x 128, so that the base model is still learning
# when we stop it, and RL runs in about a minute on the CPU.
N_LAYER, D_MODEL, N_HEAD = 2, 48, 2
CHUNK = 512


def new_model() -> GPT:
    cfg = GPTConfig(vocab_size=task.VOCAB, block_size=task.PROMPT_LEN + task.MAX_REPLY,
                    n_layer=N_LAYER, n_head=N_HEAD, d_model=D_MODEL, dropout=0.0)
    return GPT(cfg)


def prompt_ids(pairs) -> torch.Tensor:
    return torch.tensor([task.encode(task.prompt(int(a), int(b))) for a, b in pairs], dtype=torch.long)


@torch.no_grad()
def sample(model: GPT, prompts: torch.Tensor, temperature: float = 1.0,
           gen: torch.Generator | None = None, greedy: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, PROMPT_LEN) prompts -> (N, MAX_REPLY) reply ids and (N, MAX_REPLY) float mask.

    Runs in slices of CHUNK replies: one slice of 4,000 grew the process past 4 GB.
    """
    if prompts.shape[0] > CHUNK:
        parts = [sample(model, p, temperature, gen, greedy) for p in prompts.split(CHUNK)]
        return torch.cat([a for a, _ in parts]), torch.cat([m for _, m in parts])
    model.eval()
    x = prompts
    n = x.shape[0]
    done = torch.zeros(n, dtype=torch.bool)
    out = torch.full((n, task.MAX_REPLY), task.PAD, dtype=torch.long)
    mask = torch.zeros(n, task.MAX_REPLY)
    for t in range(task.MAX_REPLY):
        logits = model(x)[0][:, -1, :]
        logits[:, task.PAD] = float("-inf")              # PAD is never a target, so never sampled
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, task.PAD), nxt)
        mask[:, t] = (~done).float()
        out[:, t] = nxt
        done = done | (nxt == task.END)
        x = torch.cat([x, nxt[:, None]], 1)
        if done.all():
            break
    return out, mask


def rewards(replies: torch.Tensor, pairs_rep) -> tuple[torch.Tensor, list[str]]:
    """Checker reward per reply (0 or 1) and the reply texts. ``pairs_rep`` holds one (a, b) per reply."""
    texts = [task.reply_text(r.tolist()) for r in replies]
    r = torch.tensor([task.check(s, int(a), int(b)) for s, (a, b) in zip(texts, pairs_rep)], dtype=torch.float32)
    return r, texts


def token_logprobs(model: GPT, prompts: torch.Tensor, replies: torch.Tensor,
                   with_logits: bool = False):
    """Log-probability of each reply character under the model, (N, MAX_REPLY); 0 at PAD positions."""
    seq = torch.cat([prompts, replies], 1)
    logits = model(seq[:, :-1])[0][:, prompts.shape[1] - 1:, :]
    logits = logits.masked_fill(
        torch.arange(task.VOCAB) == task.PAD, float("-inf"))   # same support as sampling
    logp = F.log_softmax(logits, -1)
    lp = logp.gather(-1, replies[..., None]).squeeze(-1)
    lp = lp.masked_fill(replies == task.PAD, 0.0)               # -inf there; 0 keeps mask products finite
    return (lp, logp) if with_logits else lp
