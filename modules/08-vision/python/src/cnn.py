"""A tiny CNN on a FashionMNIST subset — so we can *see* the kernels it learns.

The hand-designed kernels in ``conv.py`` were invented by humans. Here we train
a two-conv-layer network by gradient descent and then look at its first-layer
filters. They arrange themselves into oriented edge / blob detectors — nobody
told them to. That is the whole idea of "learned features".
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_device

FASHION = "zalando-datasets/fashion_mnist"
FASHION_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class TinyCNN(nn.Module):
    """conv(1->8, 5x5) -> pool -> conv(8->16, 3x3) -> pool -> fc -> 10.

    Deliberately small so first-layer filters are 5x5 and human-readable.
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.fc = nn.Linear(16 * 7 * 7, n_classes)

    def features(self, x):
        """Return (post-conv1 activations, post-conv2 activations)."""
        a1 = F.relu(self.conv1(x))
        p1 = F.max_pool2d(a1, 2)
        a2 = F.relu(self.conv2(p1))
        return a1, a2

    def forward(self, x):
        a1 = F.max_pool2d(F.relu(self.conv1(x)), 2)   # -> 14x14
        a2 = F.max_pool2d(F.relu(self.conv2(a1)), 2)  # -> 7x7
        return self.fc(a2.flatten(1))


def load_fashion_subset(n_train: int = 6000, n_test: int = 1000, seed: int = 0):
    """Load a FashionMNIST subset as float tensors in [0, 1], shape (N,1,28,28)."""
    from datasets import load_dataset

    ds = load_dataset(FASHION)

    def to_tensors(split, n):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(split), size=min(n, len(split)), replace=False)
        sub = split.select(sorted(int(i) for i in idx))
        imgs = np.stack([np.asarray(im, dtype=np.float32) for im in sub["image"]])
        x = torch.from_numpy(imgs / 255.0).unsqueeze(1)
        y = torch.tensor(sub["label"], dtype=torch.long)
        return x, y

    xtr, ytr = to_tensors(ds["train"], n_train)
    xte, yte = to_tensors(ds["test"], n_test)
    return (xtr, ytr), (xte, yte)


def train_tiny_cnn(epochs: int = 6, batch_size: int = 128, lr: float = 1e-3, seed: int = 0):
    """Train TinyCNN on the subset; return (model, history, test_acc)."""
    torch.manual_seed(seed)
    device = get_device()
    (xtr, ytr), (xte, yte) = load_fashion_subset(seed=seed)
    xtr, ytr = xtr.to(device), ytr.to(device)
    xte, yte = xte.to(device), yte.to(device)

    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = xtr.shape[0]
    history = {"loss": [], "test_acc": []}

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        running = 0.0
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            logits = model(xtr[b])
            loss = F.cross_entropy(logits, ytr[b])
            loss.backward()
            opt.step()
            running += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            acc = (model(xte).argmax(1) == yte).float().mean().item()
        history["loss"].append(running / n)
        history["test_acc"].append(acc)
        print(f"epoch {ep + 1}/{epochs}  loss {running / n:.4f}  test_acc {acc:.4f}")

    return model, history, history["test_acc"][-1]


if __name__ == "__main__":
    model, hist, acc = train_tiny_cnn()
    print(f"final test accuracy: {acc:.4f}")
