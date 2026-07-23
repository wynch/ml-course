"""The training loop: mini-batch SGD/Adam over epochs, with a train/val split.

This is the beating heart of the module. Everything else (layers, losses,
optimizers) exists so that this loop can run. The loop itself is small and
worth reading line by line: shuffle, slice into mini-batches, forward, compute
loss, backward, step, repeat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nn import MLP, SoftmaxCrossEntropy, accuracy


@dataclass
class History:
    """Per-epoch metrics, ready to plot."""

    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)


def train_val_split(x, y, val_frac, rng):
    """Shuffle and split into training and validation sets."""
    n = x.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def iterate_minibatches(x, y, batch_size, rng):
    """Yield shuffled mini-batches of (x, y)."""
    n = x.shape[0]
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        b = idx[start : start + batch_size]
        yield x[b], y[b]


def train(
    model: MLP,
    optimizer,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
    loss_fn=None,
    on_epoch_end=None,
    verbose: bool = True,
) -> History:
    """Run the full training loop and return the metric history.

    ``on_epoch_end(epoch, model)`` is an optional callback used, for example,
    to snapshot the decision boundary for the animated GIF.
    """
    loss_fn = loss_fn or SoftmaxCrossEntropy()
    hist = History()

    for epoch in range(epochs):
        batch_losses = []
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, rng):
            logits = model.forward(xb)
            batch_losses.append(loss_fn.forward(logits, yb))
            model.backward(loss_fn.backward())
            optimizer.step()

        # end-of-epoch metrics
        tr_logits = model.forward(x_train)
        val_logits = model.forward(x_val)
        hist.train_loss.append(float(np.mean(batch_losses)))
        hist.val_loss.append(loss_fn.forward(val_logits, y_val))
        hist.train_acc.append(accuracy(model, x_train, y_train))
        hist.val_acc.append(accuracy(model, x_val, y_val))

        if on_epoch_end is not None:
            on_epoch_end(epoch, model)

        if verbose:
            print(
                f"epoch {epoch + 1:3d}/{epochs} | "
                f"train loss {hist.train_loss[-1]:.4f} "
                f"acc {hist.train_acc[-1]:.3f} | "
                f"val loss {hist.val_loss[-1]:.4f} "
                f"acc {hist.val_acc[-1]:.3f}"
            )

    return hist
