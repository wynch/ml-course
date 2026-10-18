"""The in-context regression task, the transformer that learns it, and four classic rules.

A prompt holds N_POINTS points of one task: w ~ N(0, I_d), x_i ~ N(0, I_d), y_i = w . x_i.
It is fed as the sequence x_1, y_1, x_2, y_2, ...; at the token x_{n+1} the net predicts
y_{n+1} from the n pairs before it, so one prompt scores every n from 0 to N_POINTS - 1.

Errors are MSE / d: predicting 0 scores 1, because Var(y) = E|w|^2 = d.
"""

from __future__ import annotations

import math

import numpy as np
import torch

D = 5
N_POINTS = 21                 # 20 context pairs + the last query
STEPS, CONTROL_STEPS, BATCH = 20_000, 10_000, 64
LR, LR_MIN, WARMUP, CLIP = 3e-4, 3e-5, 500, 1.0
N_EVAL = 1000
RIDGE_LAMBDA = 0.1
KNN_K = 3
SEED_WEIGHTS, SEED_TRAIN, SEED_EVAL, SEED_EVAL_SHUFFLED = 0, 1, 2, 3


class ICLNet(torch.nn.Module):
    """A small GPT-style transformer that reads (x, y) tokens and predicts each y.

    An x token is (x, 0) and a y token is (0, ..., 0, y), both in R^(d+1). The
    read-out at each x token is the prediction of the y that follows it.
    """

    def __init__(self, d: int = D, n_points: int = N_POINTS, d_model: int = 128, heads: int = 4,
                 d_mlp: int = 512, layers: int = 2) -> None:
        super().__init__()
        self.d = d
        self.read_in = torch.nn.Linear(d + 1, d_model)
        self.pos = torch.nn.Parameter(torch.zeros(2 * n_points, d_model))
        torch.nn.init.normal_(self.pos, 0.0, 0.02)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model, heads, d_mlp, dropout=0.0, activation="gelu",
                                             batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(d_model)
        self.read_out = torch.nn.Linear(d_model, 1)
        self.register_buffer("mask", torch.nn.Transformer.generate_square_subsequent_mask(2 * n_points),
                             persistent=False)

    def tokens(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        b, n, _ = xs.shape
        tok = xs.new_zeros(b, 2 * n, self.d + 1)
        tok[:, 0::2, : self.d] = xs
        tok[:, 1::2, self.d] = ys
        return tok

    def forward(self, xs: torch.Tensor, ys: torch.Tensor, hidden: bool = False):
        """xs (B, N, d), ys (B, N) -> predictions (B, N); with hidden=True also the
        residual stream at the x tokens after the embedding and after each block."""
        n = xs.shape[1]
        h = self.read_in(self.tokens(xs, ys)) + self.pos[: 2 * n]
        hs = [h[:, 0::2]]
        for blk in self.blocks:
            h = blk(h, src_mask=self.mask[: 2 * n, : 2 * n], is_causal=True)
            hs.append(h[:, 0::2])
        pred = self.read_out(self.norm(h[:, 0::2])).squeeze(-1)
        return (pred, hs) if hidden else pred


def sample(n_prompts: int, gen: torch.Generator, d: int = D, n_points: int = N_POINTS):
    """Fresh training prompts: (xs, ys)."""
    w = torch.randn(n_prompts, d, generator=gen)
    xs = torch.randn(n_prompts, n_points, d, generator=gen)
    return xs, torch.einsum("bnd,bd->bn", xs, w)


def eval_set(seed: int, shuffled: bool, n: int = N_EVAL, d: int = D, n_points: int = N_POINTS):
    """Held-out prompts as float64 arrays (xs, ys, w). With shuffled=True, prompt b
    carries the y values of prompt b+1: no y can be predicted from its own prompt."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, d))
    xs = rng.standard_normal((n, n_points, d))
    ys = np.einsum("bnd,bd->bn", xs, w)
    if shuffled:
        ys = np.roll(ys, -1, axis=0)
    return xs, ys, w


def lr_at(step: int, total: int) -> float:
    """Linear warm-up, then cosine from LR down to LR_MIN at the last step."""
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, total - WARMUP)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * t))


def train(net: ICLNet, steps: int, shuffled: bool, device: torch.device, log: list,
          n_points: int = N_POINTS, xs_fn=None) -> ICLNet:
    """AdamW on fresh prompts every step; the loss is the mean of (prediction - y)^2 / d.

    xs_fn, if given, maps the sampled xs before use (the line net fixes one coordinate to 1).
    """
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    gen = torch.Generator().manual_seed(SEED_TRAIN)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps)
        if xs_fn is None:
            xs, ys = sample(BATCH, gen, net.d, n_points)
        else:
            xs, ys = xs_fn(BATCH, gen)
        if shuffled:
            ys = torch.roll(ys, -1, dims=0)
        xs, ys = xs.to(device), ys.to(device)
        loss = ((net(xs, ys) - ys) ** 2).mean() / net.d
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        if (step + 1) % 100 == 0:
            log.append([step + 1, round(loss.item(), 5)])     # that batch's loss; item() syncs MPS
        if (step + 1) % 5000 == 0:
            print(f"  step {step + 1}: loss {log[-1][1]:.4f}", flush=True)
    return net.eval()


@torch.no_grad()
def net_predict(net: ICLNet, xs: np.ndarray, ys: np.ndarray, device: torch.device, chunk: int = 250) -> np.ndarray:
    out = []
    for i in range(0, len(xs), chunk):
        x = torch.from_numpy(xs[i:i + chunk]).float().to(device)
        y = torch.from_numpy(ys[i:i + chunk]).float().to(device)
        out.append(net(x, y).cpu().double().numpy())
    return np.concatenate(out)


def mse_over_d(pred: np.ndarray, ys: np.ndarray, d: int = D) -> np.ndarray:
    """MSE / d at each n, averaged over prompts."""
    return ((pred - ys) ** 2).mean(0) / d


# ---- four classic rules, each fitted on the n pairs before the query -------------------------

def w_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimum-norm least squares, defined for n < d too."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def w_ridge(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def w_one_gd_step(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One gradient step from w = 0 on (1/2) sum (w.x_i - y_i)^2, step 1/(n + d + 1).

    For Gaussian x that step minimises the expected error; the expected MSE/d is then
    (d + 1) / (n + d + 1).
    """
    n, d = X.shape
    return X.T @ y / (n + d + 1)


def knn_predict(X: np.ndarray, y: np.ndarray, xq: np.ndarray, k: int = KNN_K) -> float:
    near = np.argsort(((X - xq) ** 2).sum(1))[:k]
    return float(y[near].mean())


def baselines(xs: np.ndarray, ys: np.ndarray) -> dict[str, np.ndarray]:
    """Squared error of each rule at each (prompt, n). At n = 0 every rule predicts 0."""
    names = ("least_squares", "ridge_0.1", "knn_3", "one_gd_step")
    err = {k: np.zeros(ys.shape) for k in names}
    for b in range(len(xs)):
        for n in range(xs.shape[1]):
            xq, yq = xs[b, n], ys[b, n]
            if n == 0:
                for k in names:
                    err[k][b, n] = yq ** 2
                continue
            X, y = xs[b, :n], ys[b, :n]
            err["least_squares"][b, n] = (xq @ w_least_squares(X, y) - yq) ** 2
            err["ridge_0.1"][b, n] = (xq @ w_ridge(X, y) - yq) ** 2
            err["one_gd_step"][b, n] = (xq @ w_one_gd_step(X, y) - yq) ** 2
            err["knn_3"][b, n] = (knn_predict(X, y, xq) - yq) ** 2
    return err
