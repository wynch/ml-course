"""Section 6: the library way. Meta's flow_matching package and LLaDA's sampler, against our code.

flow_matching 1.0.10 (PyPI; the code that comes with Lipman et al.'s 2024
guide, CC BY-NC licence) has both kinds of flow this module builds:

  continuous  AffineProbPath + CondOTScheduler: the straight line of section 1;
              ODESolver: the Euler sampler.
  discrete    MixtureDiscreteProbPath with a source that is all [MASK] and
              PolynomialConvexScheduler(n=1): the masking process of section 2,
              with time running the other way (library t = 1 - our t);
              MixturePathGeneralizedKL: its loss; MixtureDiscreteEulerSolver: its sampler.

LLaDA (Nie et al., 2025) publishes its sampler as ``generate()`` in
github.com/ML-GSAI/LLaDA (read at commit 9182493, 2026-10-20). The repository
carries no licence, so the function is not copied here: ``llada_generate``
below is re-typed from it, step for step, without its chat-only options
(classifier-free guidance, end-of-text handling, the end-of-thinking boost).

    uv run python src/library.py         # writes results/library.json, about 2 minutes on the CPU
"""

from __future__ import annotations

import numpy as np

import common
import torch
import torch.nn.functional as F
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import AffineProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import CondOTScheduler, PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver, ODESolver
from flow_matching.utils import ModelWrapper

import data as D
import diffusion as Df
import flow2d
from sample import scores
from train import load_model

N = 64


# ------------------------------------------------------------------ continuous
class Velocity(ModelWrapper):
    def forward(self, x, t, **extras):
        return self.model(x, t.expand(len(x)))


def check_continuous() -> dict:
    torch.manual_seed(0)
    x1 = torch.from_numpy(flow2d.make_spiral(512, seed=0))
    x0, t = torch.randn_like(x1), torch.rand(512)
    ps = AffineProbPath(scheduler=CondOTScheduler()).sample(x_0=x0, x_1=x1, t=t)
    xt, v = flow2d.fm_target(x0, x1, t)
    model, _ = flow2d.train("fm", torch.from_numpy(flow2d.make_spiral(2000, seed=0)), steps=500)
    z = torch.randn(1000, 2, generator=torch.Generator().manual_seed(7))
    ours = flow2d.sample_fm(model, z, 16)
    lib = ODESolver(velocity_model=Velocity(model)).sample(x_init=z, step_size=1 / 16, method="euler",
                                                           time_grid=torch.tensor([0.0, 1.0]))
    return {"x_t_max_gap": float((ps.x_t - xt).abs().max()), "velocity_max_gap": float((ps.dx_t - v).abs().max()),
            "euler16_max_gap": float((lib - ours).abs().max()), "model_train_steps": 500}


# ------------------------------------------------------------------ discrete
def carry_over(logits: torch.Tensor, xt: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Logits over the 65 characters + [MASK] (never predicted), with every unmasked position
    copied through: probability 1 on the character it already shows (MDLM's carry-over)."""
    full = torch.cat([logits, torch.full_like(logits[..., :1], -torch.inf)], -1)
    keep = xt != mask_id
    onehot = torch.full_like(full, -torch.inf).scatter(-1, xt.clamp(max=mask_id)[..., None], 0.0)
    return torch.where(keep[..., None], onehot, full)


def check_loss(mdm, blocks) -> dict:
    M, V = mdm.cfg.mask_id, mdm.cfg.vocab_size
    path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
    torch.manual_seed(0)
    x1 = blocks
    t_lib = torch.rand(len(x1)) * 0.98 + 0.01
    xt = path.sample(x_0=torch.full_like(x1, M), x_1=x1, t=t_lib).x_t
    with torch.no_grad():
        logits = mdm(xt)
    masked = xt == M
    t_ours = 1 - t_lib
    ce = F.cross_entropy(logits.transpose(1, 2), x1, reduction="none")
    ours = ((ce * masked) / t_ours[:, None]).sum(1) / x1.size(1)
    kl = MixturePathGeneralizedKL(path, reduction="none")
    lib = kl(carry_over(logits, xt, M), x1, xt, t_lib).sum(1) / x1.size(1)
    full = torch.cat([logits, torch.full_like(logits[..., :1], -torch.inf)], -1)
    lib_raw = kl(full, x1, xt, t_lib).sum(1) / x1.size(1)
    return {"lines": len(x1), "mask_rate_vs_1_minus_t": round(float((masked.float().mean(1) - t_ours).abs().mean()), 4),
            "ours_mean": round(float(ours.mean()), 5), "library_mean": round(float(lib.mean()), 5),
            "max_relative_gap": float(((lib - ours).abs() / ours.clamp_min(1e-8)).max()),
            "library_without_carry_over_mean": round(float(lib_raw.mean()), 5), "vocab": V}


class Posterior(ModelWrapper):
    """flow_matching's solver wants probabilities over the whole vocabulary, [MASK] included."""

    def forward(self, x, t, **extras):
        with torch.no_grad():
            return torch.softmax(carry_over(self.model(x), x, self.model.cfg.mask_id), -1)


def check_solver(mdm, tok, vocab, judge, lu) -> dict:
    M = mdm.cfg.mask_id
    path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
    solver = MixtureDiscreteEulerSolver(model=Posterior(mdm), path=path, vocabulary_size=M + 1)
    out = {}
    for steps in (16, 128):
        torch.manual_seed(steps)
        ids = solver.sample(x_init=torch.full((N, 128), M), step_size=None, time_grid=torch.linspace(0, 1, steps + 1))
        ours = Df.sample_mdm(mdm, N, steps, "random", 1.0, gen=torch.Generator().manual_seed(steps))
        out[str(steps)] = {"library_euler": scores(ids, tok, vocab, judge, lu), "ours_random": scores(ours, tok, vocab, judge, lu),
                           "library_left_masked": int((ids == M).sum())}
    return out


# ------------------------------------------------------------------ LLaDA's sampler, re-typed
def add_gumbel_noise(logits, temperature, gen):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand(logits.shape, generator=gen, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def llada_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.0,
                   remasking="low_confidence", gen=None):
    mask_id = model.cfg.mask_id
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long)
    x[:, : prompt.shape[1]] = prompt.clone()
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    for num_block in range(num_blocks):
        lo = prompt.shape[1] + num_block * block_length
        block_mask_index = x[:, lo: lo + block_length] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            logits = model(x)
            logits_with_noise = add_gumbel_noise(logits, temperature, gen)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            else:
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), generator=gen)
            x0_p[:, lo + block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=int(num_transfer_tokens[j, i]))
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x


def check_llada(mdm, tok, vocab, judge, lu) -> dict:
    empty = torch.zeros((N, 0), dtype=torch.long)
    out = {}
    for steps in (16, 128):
        a = llada_generate(mdm, empty, steps=steps, temperature=1.0, gen=torch.Generator().manual_seed(steps))
        b = Df.sample_mdm(mdm, N, steps, "confidence", 1.0, gen=torch.Generator().manual_seed(steps))
        out[f"same_as_ours_{steps}_steps"] = round(float((a == b).float().mean()), 5)
    semi = llada_generate(mdm, empty, steps=128, block_length=16, temperature=1.0, gen=torch.Generator().manual_seed(3))
    out["semi_ar_block16_128_steps"] = scores(semi, tok, vocab, judge, lu)
    out["pure_128_steps"] = scores(llada_generate(mdm, empty, steps=128, temperature=1.0,
                                                  gen=torch.Generator().manual_seed(3)), tok, vocab, judge, lu)
    return out


def main() -> None:
    guard = common.Guard("library", "cpu")
    tok, train, val = D.load()
    vocab = D.word_set(train, tok)
    lu = D.log_unigram(train, tok.vocab_size)
    mdm, judge = load_model("mdm", "cpu"), load_model("ar", "cpu")
    res = {"continuous": check_continuous()}
    guard.tick("continuous")
    res["loss"] = check_loss(mdm, D.val_blocks(val, 128))
    guard.tick("loss")
    res["solver"] = check_solver(mdm, tok, vocab, judge, lu)
    guard.tick("solver")
    res["llada"] = check_llada(mdm, tok, vocab, judge, lu)
    guard.tick("llada")
    res["versions"] = {"flow_matching": "1.0.10", "llada_repo_commit": "9182493720ed723ef8031210d85959364e51cbe0"}
    res["guard"] = guard.report()
    common.write_results("library", res)


if __name__ == "__main__":
    main()
