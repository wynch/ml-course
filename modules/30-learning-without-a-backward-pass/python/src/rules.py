"""Six ways to train one MLP: backprop, backprop with checkpointing, MeZO
(two-point zeroth-order), forward-forward, Mono-Forward and predictive coding.

Every learner has the same interface:

    step(x, y) -> float      one update on a batch; returns the loss it used
    predict(x) -> Tensor     a score per class, shape (batch, 10)

Backprop and checkpointing use torch's autograd: that is the backward pass
this module tries to do without. The other four are written by hand, with
no autograd at all, so what each one keeps in memory is exactly what the
code holds. Each learner uses Adam (written once, below), except MeZO, which
keeps no optimizer state on purpose.

Shapes: rows are examples. A layer computes z = a W^T + b, with W of shape
(out, in).
"""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

N_CLASSES = 10


# ------------------------------------------------------------------ pieces
def he_layer(n_in: int, n_out: int, gen: torch.Generator) -> list[torch.Tensor]:
    """He-normal weights, zero bias (module 02's init)."""
    w = torch.randn(n_out, n_in, generator=gen) * math.sqrt(2.0 / n_in)
    return [w, torch.zeros(n_out)]


class Adam:
    """Adam on a list of tensors, with the gradients passed in. Two extra
    numbers per parameter: the running mean and the running square."""

    def __init__(self, params: list[torch.Tensor], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, betas[0], betas[1], eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    @torch.no_grad()
    def step(self, grads: list[torch.Tensor]) -> None:
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.addcdiv_(m / c1, (v / c2).sqrt_().add_(self.eps), value=-self.lr)


class SGD:
    """Plain gradient descent, with the same interface as Adam: no state at all."""

    def __init__(self, params: list[torch.Tensor], lr: float = 1e-2):
        self.params, self.lr = params, lr

    @torch.no_grad()
    def step(self, grads: list[torch.Tensor]) -> None:
        for p, g in zip(self.params, grads):
            p.sub_(g, alpha=self.lr)


def make_opt(kind: str, params: list[torch.Tensor], lr: float):
    return {"adam": Adam, "sgd": SGD}[kind](params, lr)


def softmax_ce(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Mean cross-entropy and its gradient with respect to the logits."""
    p = torch.softmax(logits, dim=1)
    loss = -torch.log(p[torch.arange(len(y)), y].clamp_min(1e-12)).mean()
    g = p
    g[torch.arange(len(y)), y] -= 1.0
    return float(loss), g / len(y)


# ------------------------------------------------------- backprop (autograd)
class Backprop:
    """The control: autograd computes every gradient in one backward pass.
    ``segments > 0`` turns on gradient checkpointing: the hidden blocks are
    split into that many segments, only each segment's input is kept, and the
    segment's forward pass runs again during the backward pass (Chen et al.,
    2016). Same numbers, less memory, more compute."""

    name = "backprop"

    def __init__(self, sizes, seed: int, lr: float = 1e-3, segments: int = 0, opt: str = "adam"):
        gen = torch.Generator().manual_seed(seed)
        self.layers = [he_layer(a, b, gen) for a, b in zip(sizes[:-1], sizes[1:])]
        self.params = [t.requires_grad_() for wb in self.layers for t in wb]
        self.opt = make_opt(opt, self.params, lr)
        self.segments = segments
        if segments:
            self.name = "checkpointing"

    def _block(self, i: int, a: torch.Tensor) -> torch.Tensor:
        w, b = self.layers[i]
        return torch.relu(a @ w.T + b)

    def _run(self, lo: int, hi: int, a: torch.Tensor) -> torch.Tensor:
        for i in range(lo, hi):
            a = self._block(i, a)
        return a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_hidden = len(self.layers) - 1
        a = x
        if self.segments and torch.is_grad_enabled():
            edges = [round(k * n_hidden / self.segments) for k in range(self.segments + 1)]
            for lo, hi in zip(edges[:-1], edges[1:]):
                a = checkpoint(self._run, lo, hi, a, use_reentrant=False)
        else:
            a = self._run(0, n_hidden, a)
        w, b = self.layers[-1]
        return a @ w.T + b

    def step(self, x, y) -> float:
        loss = torch.nn.functional.cross_entropy(self.forward(x), y)
        grads = torch.autograd.grad(loss, self.params)
        self.opt.step(grads)
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x)


# ------------------------------------------------------------- MeZO (SPSA)
class MeZO:
    """Two forward passes along one random direction z per step:

        g = (L(θ + εz) − L(θ − εz)) / 2ε,      θ ← θ − lr · g · z

    z is never stored: it is drawn again from its seed whenever it is needed,
    and the weights are nudged in place (Malladi et al., 2023). Memory is the
    weights plus one forward pass. No gradient, no optimizer state."""

    name = "mezo"

    def __init__(self, sizes, seed: int, lr: float = 1e-3, eps: float = 1e-3):
        gen = torch.Generator().manual_seed(seed)
        self.layers = [he_layer(a, b, gen) for a, b in zip(sizes[:-1], sizes[1:])]
        self.params = [t for wb in self.layers for t in wb]
        self.lr, self.eps = lr, eps
        self.seeds = torch.Generator().manual_seed(10_000 + seed)

    def forward(self, x):
        a = x
        for w, b in self.layers[:-1]:
            a = torch.relu(a @ w.T + b)
        w, b = self.layers[-1]
        return a @ w.T + b

    def _nudge(self, seed: int, scale: float) -> None:
        g = torch.Generator().manual_seed(seed)
        for p in self.params:
            p.add_(torch.randn(p.shape, generator=g), alpha=scale)

    def loss(self, x, y) -> float:
        return float(torch.nn.functional.cross_entropy(self.forward(x), y))

    @torch.no_grad()
    def projected_grad(self, x, y, seed: int) -> tuple[float, float]:
        """The two-point estimate along z(seed); leaves the weights as found."""
        self._nudge(seed, +self.eps)
        lp = self.loss(x, y)
        self._nudge(seed, -2 * self.eps)
        lm = self.loss(x, y)
        self._nudge(seed, +self.eps)
        return (lp - lm) / (2 * self.eps), (lp + lm) / 2

    @torch.no_grad()
    def step(self, x, y) -> float:
        seed = int(torch.randint(2**62, (1,), generator=self.seeds))
        g, loss = self.projected_grad(x, y, seed)
        self._nudge(seed, -self.lr * g)
        return loss

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x)


# --------------------------------------------------------- forward-forward
def overlay(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Write the label into the first 10 pixels as a one-hot (Hinton, 2022)."""
    x = x.clone()
    x[:, :N_CLASSES] = 0.0
    x[torch.arange(len(x)), labels] = 1.0
    return x


def normalise(a: torch.Tensor) -> torch.Tensor:
    """Divide each row by its length, so the next layer cannot read the
    goodness of this one from the size of its input, only from its direction."""
    return a / (a.norm(dim=1, keepdim=True) + 1e-6)


def ff_layer_grads(w, b, x_pos, x_neg, theta: float):
    """One forward-forward layer: goodness g = mean of a² over units. The loss
    asks g > θ for real (positive) data and g < θ for wrong-label (negative)
    data: softplus(θ − g_pos) + softplus(g_neg − θ), averaged over the batch.
    Returns the loss, the gradients for w and b, and both outputs."""
    out, grads_w, grads_b, loss = [], 0.0, 0.0, 0.0
    for x, sign in ((x_pos, -1.0), (x_neg, +1.0)):
        z = x @ w.T + b
        a = torch.relu(z)
        g = (a * a).mean(dim=1)
        s = sign * (g - theta)                               # positive: θ − g; negative: g − θ
        loss += float(torch.nn.functional.softplus(s).mean())
        dg = torch.sigmoid(s) * sign / len(x)                # d loss / d g
        dz = dg[:, None] * 2 * a / a.shape[1] * (z > 0)
        grads_w = grads_w + dz.T @ x
        grads_b = grads_b + dz.sum(0)
        out.append(a)
    return loss, grads_w, grads_b, out[0], out[1]


class ForwardForward:
    """Each hidden layer learns on its own to give high goodness to an image
    with its right label written in, and low goodness to the same image with a
    wrong label. No layer sends anything backwards. To classify, try all ten
    labels and pick the one with the highest total goodness."""

    name = "forward_forward"

    def __init__(self, sizes, seed: int, lr: float = 1e-3, theta: float = 2.0):
        gen = torch.Generator().manual_seed(seed)
        hidden = sizes[:-1]                                  # the 10-way output layer is not used
        self.layers = [he_layer(a, b, gen) for a, b in zip(hidden[:-1], hidden[1:])]
        self.opts = [Adam(wb, lr) for wb in self.layers]
        self.theta = theta
        self.neg = torch.Generator().manual_seed(20_000 + seed)

    @torch.no_grad()
    def step(self, x, y) -> float:
        wrong = (y + torch.randint(1, N_CLASSES, y.shape, generator=self.neg)) % N_CLASSES
        h_pos, h_neg = overlay(x, y), overlay(x, wrong)
        total = 0.0
        for (w, b), opt in zip(self.layers, self.opts):
            loss, gw, gb, a_pos, a_neg = ff_layer_grads(w, b, h_pos, h_neg, self.theta)
            opt.step([gw, gb])
            total += loss
            h_pos, h_neg = normalise(a_pos), normalise(a_neg)
        return total / len(self.layers)

    @torch.no_grad()
    def predict(self, x):
        scores = torch.zeros(len(x), N_CLASSES)
        for c in range(N_CLASSES):
            h = overlay(x, torch.full((len(x),), c))
            for w, b in self.layers:
                a = torch.relu(h @ w.T + b)
                scores[:, c] += (a * a).mean(dim=1)
                h = normalise(a)
        return scores


# ------------------------------------------------------------- Mono-Forward
def mf_layer_grads(w, b, m, x, y):
    """One Mono-Forward layer: a = relu(x Wᵀ + b), local logits a Mᵀ, local
    cross-entropy. The gradient stops at x: nothing flows to earlier layers."""
    z = x @ w.T + b
    a = torch.relu(z)
    loss, dlog = softmax_ce(a @ m.T, y)
    gm = dlog.T @ a
    dz = (dlog @ m) * (z > 0)
    return loss, dz.T @ x, dz.sum(0), gm, a


class MonoForward:
    """Each hidden layer has its own small 10-way head M_i and minimises its
    own cross-entropy (Gong et al., 2025). The output is the last layer's head."""

    name = "mono_forward"

    def __init__(self, sizes, seed: int, lr: float = 1e-3, opt: str = "adam"):
        gen = torch.Generator().manual_seed(seed)
        hidden = sizes[:-1]
        self.layers = [he_layer(a, b, gen) for a, b in zip(hidden[:-1], hidden[1:])]
        self.heads = [torch.randn(N_CLASSES, n, generator=gen) * math.sqrt(1.0 / n) for n in hidden[1:]]
        self.opts = [make_opt(opt, [w, b, m], lr) for (w, b), m in zip(self.layers, self.heads)]

    @torch.no_grad()
    def step(self, x, y) -> float:
        a, total = x, 0.0
        for (w, b), m, opt in zip(self.layers, self.heads, self.opts):
            loss, gw, gb, gm, a = mf_layer_grads(w, b, m, a, y)
            opt.step([gw, gb, gm])
            total += loss
        return total / len(self.layers)

    @torch.no_grad()
    def predict(self, x, layer: int = -1):
        a = x
        outs = []
        for (w, b), m in zip(self.layers, self.heads):
            a = torch.relu(a @ w.T + b)
            outs.append(a @ m.T)
        return outs[layer]


# ------------------------------------------------------- predictive coding
class PredictiveCoding:
    """Supervised predictive coding (Whittington and Bogacz, 2017; Millidge et
    al., 2020). Each hidden layer holds a value x_l that may differ from its
    prediction μ_l = φ(x_{l−1}) W_lᵀ + b_l. The energy

        F = Σ_hidden ½‖x_l − μ_l‖² + CE(μ_L, y)

    is lowered in two phases. Inference: the x_l slide downhill for T steps,
    with the input clamped and the label pulling on the top, so the error at
    the top spreads down, one layer per step. Learning: each W_l moves to
    shrink its own layer's error, using only the values on either side of it.

    φ is the identity on the input and ReLU elsewhere, so a pass with x_l = μ_l
    is the same function as the backprop MLP."""

    name = "predictive_coding"

    def __init__(self, sizes, seed: int, lr: float = 1e-3, T: int = 20, gamma: float = 0.1):
        gen = torch.Generator().manual_seed(seed)
        self.layers = [he_layer(a, b, gen) for a, b in zip(sizes[:-1], sizes[1:])]
        self.opt = Adam([t for wb in self.layers for t in wb], lr)
        self.T, self.gamma = T, gamma

    @staticmethod
    def phi(l: int, x: torch.Tensor) -> torch.Tensor:
        return x if l == 0 else torch.relu(x)

    def mu(self, l: int, x_prev: torch.Tensor) -> torch.Tensor:
        """Prediction for node l (1-based) from node l − 1."""
        w, b = self.layers[l - 1]
        return self.phi(l - 1, x_prev) @ w.T + b

    @torch.no_grad()
    def feedforward(self, x):
        xs = [x]
        for l in range(1, len(self.layers) + 1):
            xs.append(self.mu(l, xs[-1]))
        return xs

    @torch.no_grad()
    def deltas(self, xs, y):
        """δ_l = ∂F/∂μ_l: −(x_l − μ_l) on hidden layers, softmax − onehot at the top
        (per example; the batch mean comes in the weight gradient)."""
        L = len(self.layers)
        out = [None]
        for l in range(1, L):
            out.append(-(xs[l] - self.mu(l, xs[l - 1])))
        p = torch.softmax(self.mu(L, xs[L - 1]), dim=1)
        p[torch.arange(len(y)), y] -= 1.0
        out.append(p)
        return out

    @torch.no_grad()
    def energy(self, xs, y) -> float:
        L = len(self.layers)
        e = sum(0.5 * float(((xs[l] - self.mu(l, xs[l - 1])) ** 2).sum()) for l in range(1, L))
        logits = self.mu(L, xs[L - 1])
        return e + float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))

    @torch.no_grad()
    def infer(self, x, y, T: int | None = None, trace: list | None = None):
        """T steps of x_l ← x_l − γ ∂F/∂x_l on the hidden nodes, from the
        feedforward pass. ∂F/∂x_l = (x_l − μ_l) + φ'(x_l) ⊙ (δ_{l+1} W_{l+1})."""
        T = self.T if T is None else T
        xs = self.feedforward(x)[:-1]                          # nodes 0..L−1; the top is the loss
        L = len(self.layers)
        for _ in range(T):
            d = self.deltas(xs, y)
            for l in range(1, L):
                w_next = self.layers[l][0]
                grad = -d[l] + (xs[l] > 0) * (d[l + 1] @ w_next)
                xs[l] = xs[l] - self.gamma * grad
            if trace is not None:
                trace.append(self.weight_grads(xs, y))
        return xs

    @torch.no_grad()
    def weight_grads(self, xs, y):
        """∂F/∂W_l = δ_lᵀ φ(x_{l−1}) / batch, ∂F/∂b_l = mean δ_l: each uses only layer l and l − 1."""
        d = self.deltas(xs, y)
        n = len(y)
        out = []
        for l in range(1, len(self.layers) + 1):
            out += [d[l].T @ self.phi(l - 1, xs[l - 1]) / n, d[l].sum(0) / n]
        return out

    @torch.no_grad()
    def step(self, x, y) -> float:
        xs = self.infer(x, y)
        self.opt.step(self.weight_grads(xs, y))
        return self.energy(xs, y) / len(y)

    @torch.no_grad()
    def predict(self, x):
        return self.feedforward(x)[-1]


SIZES = (784, 256, 256, 256, 10)                 # four weight layers: three hidden of 256, one 10-way output

RULES = {
    "backprop": lambda seed, **kw: Backprop(SIZES, seed, **kw),
    "checkpointing": lambda seed, **kw: Backprop(SIZES, seed, segments=3, **kw),
    "mezo": lambda seed, **kw: MeZO(SIZES, seed, **kw),
    "forward_forward": lambda seed, **kw: ForwardForward(SIZES, seed, **kw),
    "mono_forward": lambda seed, **kw: MonoForward(SIZES, seed, **kw),
    "predictive_coding": lambda seed, **kw: PredictiveCoding(SIZES, seed, **kw),
}


def accuracy(learner, x, y, batch: int = 2000) -> float:
    right = 0
    for i in range(0, len(x), batch):
        right += int((learner.predict(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum())
    return right / len(x)
